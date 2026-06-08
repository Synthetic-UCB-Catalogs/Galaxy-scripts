"""
Executes the main `gwg` iteration loop to identify and subtract resolvable sources.

This script first merges all waveform data chunks from the previous step into a single
dataset. It then iteratively identifies the brightest sources, subtracts them, and
adds them to a catalog of resolvable sources until a stopping condition is met.

Usage:
    python main_loop.py --code <simulation_code>

Example:
    python main_loop.py --code COSMIC
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml

import os
import sys
import shutil
import argparse
import glob
import time

# Science packages
import gwg
from gbgpu.gbgpu import GBGPU
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models

from helpers import Constants, apply_global_plot_settings, load_and_prepare_config

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
    parser.add_argument('--snr-cutoff', type=float, required=True,
                        help='SNR cutoff for the main loop (REQUIRED; no config.yaml default, '
                             'so a stale cutoff can never be silently reused). Written to '
                             'icloop_kwargs.snr_cutoff and used as the per-run output suffix.')
    parser.add_argument('--datapath', type=str, required=True,
                        help='catalog subpath (REQUIRED; no config.yaml default). Must match the '
                             'datapath gen_waveforms used so the cached waveforms are found.')
    args = parser.parse_args()
    code = args.code

    config = load_and_prepare_config('config.yaml')
    config['datapath'] = args.datapath
    config['icloop_kwargs']['snr_cutoff'] = args.snr_cutoff
    with open('plot_config.yaml', 'r') as f:
        plot_settings = yaml.safe_load(f)
        
    apply_global_plot_settings(plot_settings)

    wavepath = os.path.join(config['waveformpath'], config['datapath'])
    outpath = os.path.join(config['outputpath'], config['datapath'])
    os.makedirs(outpath, exist_ok=True)
    # Suffix per-run outputs with the SNR cutoff so runs at different snr_cutoff
    # don't overwrite each other in the shared output folder.
    snr_cutoff = config['icloop_kwargs']['snr_cutoff']   # always set from the required --snr-cutoff
    run_tag = f"snr{snr_cutoff:g}"
    # Snapshot the EFFECTIVE config (including the --snr-cutoff value) for the record.
    with open(os.path.join(outpath, f'{code}_run_config_{run_tag}.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # --- Step 1: Clean Up Old Output Files ---
    print("INFO: Cleaning up old output files from previous runs...")
    # Clean up the main output .h5 file
    final_output_path = os.path.join(outpath, f'{code}_output_cat_{run_tag}.h5')
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    
    plot_pattern_to_clean = os.path.join(outpath, f'{code}_{run_tag}_iter*.pdf')
    old_plots = glob.glob(plot_pattern_to_clean)
    if old_plots:
        print(f"INFO: Removing {len(old_plots)} old iteration plots from the output directory...")
        for old_plot in old_plots:
            os.remove(old_plot)

    # --- Step 2: Find and Merge Input Chunks ---
    input_chunk_files = sorted(glob.glob(os.path.join(wavepath, f'{code}_waveform_cat_*.h5')))
    if not input_chunk_files:
        print(f"FATAL: No waveform chunk files found matching '{code}_waveform_cat_*.h5' in {wavepath}")
        sys.exit(1)
    
    print(f"INFO: Found {len(input_chunk_files)} input waveform chunks. Starting merge...")

    # Load the first chunk to initialize the data structures
    merged_cat_list = [gwg.utils.load_h5(input_chunk_files[0], key="cat")]
    merged_tdi = gwg.utils.load_h5(input_chunk_files[0], key="tdi")
    # numpy 2.x marks pandas .T.values views read-only; force a writable copy
    merged_tdi = {k: np.array(v) for k, v in merged_tdi.items()}

    # Loop through the rest of the chunks and aggregate
    for f in input_chunk_files[1:]:
        merged_cat_list.append(gwg.utils.load_h5(f, key="cat"))
        tdi_chunk = gwg.utils.load_h5(f, key="tdi")
        tdi_chunk = {k: np.array(v) for k, v in tdi_chunk.items()}
        for k in ["A", "E", "T"]:
            merged_tdi[k] += tdi_chunk[k]
    
    # Combine the catalogs into a single large array
    merged_cat = np.hstack(merged_cat_list)
    print(f"INFO: Merge complete. Final catalog has {len(merged_cat)} sources.")

    # --- Step 2: Prepare for Main Loop ---
    duration = Constants.yr * float(config['duration'])
    dt = float(config['dt'])
    df = 1/duration
    ndata = int(duration/dt)
    F = df * int((ndata+1)/2)
    fvec = np.arange(0, F, df)
    sens_mat = AET1SensitivityMatrix(fvec[1:], model=lisa_models.scirdv1, return_type='PSD')
    lisa_noise = {'f': fvec[1:], 'A': sens_mat[0], 'E': sens_mat[1], 'T': sens_mat[2]}

    # Determine if we should run on GPU or CPU
    use_gpu = config.get('use_gpu', False)
    if use_gpu:
        try:
            import cupy as cp
            # Use GPU 0 by default for this single-process task
            cp.cuda.Device(0).use()
            print("INFO: CuPy found. Running main loop on GPU 0.")
        except (ModuleNotFoundError, RuntimeError) as e:
            print(f"FATAL: 'use_gpu' is True in config, but cannot initialize GPU. Error: {e}")
            sys.exit(1)
    else:
        print("INFO: Running main loop on CPU.")

    # Initialize science objects for the chosen mode (GPU or CPU)
    orbits = lisa_models.EqualArmlengthOrbits(use_gpu=use_gpu)
    GB = GBGPU(orbits=orbits, use_gpu=use_gpu)

    icloop_kwargs = config['icloop_kwargs'].copy()
    icloop_kwargs['snr_thresh2'] = float(icloop_kwargs.pop('snr_cutoff', 7))**2
    icloop_kwargs['deltaf'] = int(icloop_kwargs.pop('window_size', 1000))
    icloop_kwargs['use_gbgpu'] = use_gpu
    icloop_kwargs['tdi2'] = config.get('tdi2', True)
    if icloop_kwargs.get('doplot', False):
        icloop_kwargs['tag'] = f"{code}_{run_tag}_"
    if icloop_kwargs.get('extra_smooth', False):
        icloop_kwargs['order'] = int(icloop_kwargs.get('order', 20000))

    # --- Step 3: Run the Main `icloop` Computation ---
    print("INFO: Starting the main `icloop` iterative process...")
    start_time = time.time()

    AET, S1, S1r, final_cat = gwg.icloop(merged_tdi, GB, merged_cat, lisa_noise, **icloop_kwargs)
    
    end_time = time.time()
    print(f"INFO: `icloop` finished in {end_time - start_time:.2f} seconds.")

    # --- Step 4: Finalize Catalog and Save Data ---
    print("INFO: Finalizing catalog: swapping 'snr2' column for 'snr'...")
    old_dtype = final_cat.dtype
    old_names = list(old_dtype.names)
    new_names = [name if name != 'snr2' else 'snr' for name in old_names]
    new_dtype = np.dtype([(new_names[i], old_dtype[i]) for i in range(len(old_dtype))])

    final_cat_with_snr = np.empty(final_cat.shape, dtype=new_dtype)
    for name in old_names:
        if name != 'snr2':
            final_cat_with_snr[name] = final_cat[name]
    
    final_cat_with_snr['snr'] = np.sqrt(final_cat['snr2'])
    print(f"INFO: Final resolved catalog has {len(final_cat_with_snr)} sources.")

    final_output_path = os.path.join(outpath, f'{code}_output_cat_{run_tag}.h5')
    gwg.utils.to_h5(final_output_path, cat=final_cat_with_snr, tdi=AET, S=S1)
    print(f"INFO: Final merged data saved to {final_output_path}")

    # --- Step 5: Generate Final Plot ---
    # Move any generated PDF files to the output directory, overwriting if they exist
    if icloop_kwargs.get('doplot', False):
        plot_pattern_to_move = f"{icloop_kwargs['tag']}iter*.pdf"
        newly_created_plots = glob.glob(plot_pattern_to_move)
        if newly_created_plots:
            print(f"INFO: Moving {len(newly_created_plots)} new iteration plots to output directory...")
            for pdf_file in newly_created_plots:
                shutil.move(pdf_file, os.path.join(outpath, os.path.basename(pdf_file)))
            
    print("INFO: Generating final summary plot...")
    fig, ax = plt.subplots(figsize=(10,8))
    # Use the frequency axis from the final output for plotting
    f_plot = S1['A'].f
    instr_noise = gwg.get_instr_noise(lisa_noise, f_plot)

    channel = plot_settings['channel_toplot']
    ax.loglog(f_plot, np.absolute(S1[channel]), label='Final Smoothed Spectrum', color='blue', lw=2)
    ax.loglog(
        instr_noise["f"], np.absolute(instr_noise[channel]), 
        'k', lw=2, label='LISA Noise', linestyle='dashed'
        )
    
    ax.legend(loc='upper left')
    ax.set_xlim(1e-4, 1e-2)
    ax.set_ylim(1e-43, 5e-40)
    ax.set_ylabel(r'Strain Spectral Density [$1/\sqrt{\mathrm{Hz}}$]')
    ax.set_xlabel(r'Frequency [$\mathrm{Hz}$]')
    ax.grid(True, linestyle=':', linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig.tight_layout()
    plot_filename = os.path.join(outpath, f'{code}_{run_tag}_tdi_noise.png')
    fig.savefig(plot_filename, dpi=plot_settings['dpi'])
    print(f"INFO: Final plot saved to {plot_filename}")

    print("\nSUCCESS: Main loop and finalization complete.")