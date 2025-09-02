"""
Executes the main `gwg` iteration loop to identify and subtract resolvable sources.

This script takes the generated LISA waveform data (AET channels) and iteratively
identifies the brightest sources, subtracts them from the data, and adds them to a
catalog of resolvable sources. The process continues until one of the following is true:
    - no more sources can be resolved above a given signal-to-noise ratio threshold;
    - the maximum number of iterations is reached;
    - the tolerance for the change in the catalog is reached.

Usage:
    python main_loop.py --code <simulation_code>

Example:
    python main_loop.py --code COSMIC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import yaml

import os
import sys
import argparse
import shutil
import glob
import time
import traceback

import multiprocessing as mp

import gwg
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models

from helpers import Constants, apply_global_plot_settings, load_and_prepare_config, format_bytes

# ==============================================================================
# WORKER FUNCTION
# ==============================================================================
def process_icloop_chunk(task_info):
    """
    Worker function to run the icloop on a single chunk of waveform data.
    """
    chunk_idx, input_filepath, config, lisa_noise, device_id = task_info
    
    try:
        use_gpu = (device_id is not None)

        if use_gpu:
            try:
                import cupy as cp
                cp.cuda.Device(device_id).use()
                print(f"INFO: Processing chunk {chunk_idx + 1} on GPU {device_id}...")
            except ModuleNotFoundError:
                print("FATAL: Worker in GPU mode could not find 'cupy'.")
                return None
        else:
            print(f"INFO: Processing chunk {chunk_idx + 1} on CPU...")

        # Load the input data for this chunk
        loaded_cat = gwg.utils.load_h5(input_filepath, key="cat")
        loaded_tdi = gwg.utils.load_h5(input_filepath, key="tdi")

        # Initialize science objects within the worker
        orbits = lisa_models.EqualArmlengthOrbits(use_gpu=use_gpu)
        GB = GBGPU(orbits=orbits, use_gpu=use_gpu)

        icloop_kwargs = config['icloop_kwargs'].copy()
        icloop_kwargs['snr_thresh2'] = float(icloop_kwargs.pop('snr_cutoff', 7))**2
        icloop_kwargs['deltaf'] = int(icloop_kwargs.pop('window_size', 1000))
        
        # --- Perform the core computation ---
        AET, S1, S1r, cat = gwg.icloop(loaded_tdi, GB, loaded_cat, lisa_noise, **icloop_kwargs)

        # --- Save intermediate output for this chunk ---
        outpath = os.path.join(config['outputpath'], config['datapath'])
        intermediate_filename = os.path.join(outpath, f"{config['code']}_output_chunk_{chunk_idx}.h5")
        
        gwg.utils.to_h5(intermediate_filename, cat=cat, tdi=AET, S=S1)
        
        print(f"INFO: Chunk {chunk_idx + 1} successfully saved to {intermediate_filename}")
        return intermediate_filename

    except Exception as e:
        print(f"!!! ERROR processing chunk {chunk_idx + 1} !!!")
        print(traceback.format_exc())
        return None
# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
    args = parser.parse_args()
    code = args.code

    config = load_and_prepare_config('config.yaml')
    config['code'] = code
    with open('plot_config.yaml', 'r') as f:
        plot_settings = yaml.safe_load(f)
        
    apply_global_plot_settings(plot_settings)

    wavepath = os.path.join(config['waveformpath'], config['datapath'])
    outpath = os.path.join(config['outputpath'], config['datapath'])
    os.makedirs(outpath, exist_ok=True)

    # --- Find Input Chunks and Clean Up Old Intermediate Files ---
    input_chunk_files = sorted(glob.glob(os.path.join(wavepath, f'{code}_waveform_cat_*.h5')))
    if not input_chunk_files:
        print(f"FATAL: No waveform chunk files found matching '{code}_waveform_cat_*.h5' in {wavepath}")
        sys.exit(1)
    
    num_chunks = len(input_chunk_files)
    print(f"INFO: Found {num_chunks} input waveform chunks to process.")

    # Clean up intermediate files from any previous failed runs
    for old_file in glob.glob(os.path.join(outpath, f'{code}_output_chunk_*.h5')):
        os.remove(old_file)

    # --- Prepare Shared Data ---
    duration = Constants.yr * float(config['duration'])
    dt = float(config['dt'])
    df = 1/duration
    ndata = int(duration/dt)
    F = df * int((ndata+1)/2)
    fvec = np.arange(0, F, df)
    sens_mat = AET1SensitivityMatrix(fvec[1:], model=lisa_models.scirdv1, return_type='PSD')
    lisa_noise = {'f': fvec[1:], 'A': sens_mat[0], 'E': sens_mat[1], 'T': sens_mat[2]}

    start_time = time.time()
    
    # --- Execute Processing ---
    tasks = [(i, input_chunk_files[i], config, lisa_noise, None) for i in range(num_chunks)]
    intermediate_files = []

    if config['use_gpu']:
        try:
            import cupy as cp
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus == 0: raise RuntimeError("No GPUs found, but use_gpu is True.")
            print(f"INFO: Found {num_gpus} GPUs. Starting parallel processing...")
        except (ModuleNotFoundError, RuntimeError) as e:
            print(f"FATAL: Could not get GPU count. Exiting. Error: {e}")
            sys.exit(1)
        
        # Assign a GPU to each task, cycling through available GPUs
        tasks = [(i, input_chunk_files[i], config, lisa_noise, i % num_gpus) for i in range(num_chunks)]
        
        with mp.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
            intermediate_files = pool.map(process_icloop_chunk, tasks)
    else:
        print("INFO: Starting sequential processing on CPU...")
        for task in tasks:
            intermediate_files.append(process_icloop_chunk(task))

    end_time = time.time()
    print(f"\nINFO: All chunks processed in {end_time - start_time:.2f} seconds.")

    # --- Finalization and Merging Step ---
    if None in intermediate_files:
        print("ERROR: One or more chunks failed to process. Aborting final merge.")
        sys.exit(1)

    print("\nINFO: Starting finalization: merging all chunk data...")

    # --- THIS IS THE CRITICAL FIX (PART 1) ---
    # Load the first chunk to initialize data structures AND to capture the
    # "ground truth" frequency axes. The icloop may alter the frequency grid,
    # so we cannot rely on the initial lisa_noise frequencies.
    first_chunk_file = intermediate_files[0]
    
    # Load the catalog list
    final_cat_list = [gwg.utils.load_h5(first_chunk_file, key="cat")]

    # Load the first chunk's tdi and S to start the summation and capture the axes
    tdi_from_first_chunk = gwg.utils.load_h5(first_chunk_file, key="tdi")
    S_from_first_chunk = gwg.utils.load_h5(first_chunk_file, key="S")

    # Capture the frequency axes from the first processed chunk. This is robust
    # to any changes (e.g., interpolation) made by the icloop function.
    # We treat tdi and S separately as requested for maximum robustness.
    tdi_freq_axis = tdi_from_first_chunk['f']
    S_freq_axis = S_from_first_chunk['f']
    print("INFO: Captured final frequency axes from the first processed chunk.")

    # Initialize the final summation dictionaries using a copy of the first chunk's data
    final_tdi = tdi_from_first_chunk.copy()
    final_S = S_from_first_chunk.copy()
    instr_noise = gwg.get_instr_noise(lisa_noise, S_freq_axis)
    for k in ["A", "E", "T"]:
        final_S[k] = final_S[k] - instr_noise[k]
    # --- END OF FIX (PART 1) ---

    # Loop through the rest of the chunks and aggregate
    print(f"INFO: Aggregating data from the remaining {len(intermediate_files) - 1} chunks...")
    for f in intermediate_files[1:]:
        final_cat_list.append(gwg.utils.load_h5(f, key="cat"))
        
        tdi_chunk = gwg.utils.load_h5(f, key="tdi")
        S_chunk = gwg.utils.load_h5(f, key="S")
        
        for k in ["A", "E", "T"]:
            final_tdi[k] = final_tdi[k] + tdi_chunk[k]
            final_S[k] = final_S[k] + (S_chunk[k] - instr_noise[k])

    # --- THIS IS THE CRITICAL FIX (PART 2) ---
    # After summation, the objects are plain xarrays. We must re-wrap them
    # into FrequencySeries objects using the frequency axes we captured earlier.
    print("INFO: Restoring FrequencySeries objects with correct frequency axes...")
    for k in ["A", "E", "T"]:
        final_tdi[k] = gwg.utils.FrequencySeries(final_tdi[k].data, fs=tdi_freq_axis)
        final_S[k] = final_S[k] + instr_noise[k]
        final_S[k] = gwg.utils.FrequencySeries(final_S[k].data, fs=S_freq_axis)
    # --- END OF FIX (PART 2) ---
    
    # Combine the catalogs
    final_cat = np.hstack(final_cat_list)
    
    # Perform requested SNR column transformation
    print("INFO: Swapping 'snr2' column for 'snr'...")
    old_dtype = final_cat.dtype
    old_names = list(old_dtype.names)
    new_names = [name if name != 'snr2' else 'snr' for name in old_names]
    new_dtype = np.dtype([(new_names[i], old_dtype[i]) for i in range(len(old_dtype))])

    final_cat_with_snr = np.empty(final_cat.shape, dtype=new_dtype)
    for name in old_names:
        if name != 'snr2':
            final_cat_with_snr[name] = final_cat[name]
    
    final_cat_with_snr['snr'] = np.sqrt(final_cat['snr2'])
    print(f"INFO: Final catalog has {len(final_cat_with_snr)} sources.")

    # --- Save Final Merged Data ---
    final_output_path = os.path.join(outpath, f'{code}_output_cat.h5')
    gwg.utils.to_h5(final_output_path, cat=final_cat_with_snr, tdi=final_tdi, S=final_S)
    print(f"INFO: Final merged data saved to {final_output_path}")

    # --- Clean Up Intermediate Files ---
    print("INFO: Cleaning up intermediate chunk files...")
    for f in intermediate_files:
        os.remove(f)

    # --- Generate Final Plot ---
    print("INFO: Generating final plot...")
    fig, ax = plt.subplots(figsize=(10,8))
    A = final_tdi['A'].data
    E = final_tdi['E'].data
    T = final_tdi['T'].data
    f = np.absolute(final_tdi["A"].f.to_numpy())
    df = f[1] - f[0]

    #ax.loglog(f, 2*df*np.absolute(A)**2, label='TDI A', color='green', lw=4, linestyle='--')
    ax.loglog(f, 2*df*np.absolute(E)**2, label='TDI E', color='blue', lw=1, linestyle='--')
    ax.loglog(f, final_S['E'].data, label='smoothed E', color='blue', lw=2, linestyle='solid')
    #ax.loglog(f, 2*df*np.absolute(T)**2, label='TDI T', color='red', lw=4, linestyle='--')
    ax.loglog(lisa_noise["f"], lisa_noise["E"], 'k', lw=2, label='E noise', linestyle='dashed')
    #ax.loglog(lisa_noise["f"], lisa_noise["T"], 'dimgrey', lw=1, label='T noise')
    ax.legend(loc='upper left')
    ax.set_xlim(1e-4, 1e-2)
    ax.set_ylim(1e-43, 1e-39)
    ax.set_ylabel(r'[$1/\mathrm{Hz}$]')
    ax.set_xlabel(r'Frequency [$\mathrm{Hz}$]')
    ax.grid(True, linestyle=':', linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig.tight_layout()
    plot_filename = os.path.join(outpath, f'{code}_tdi_noise.png')
    fig.savefig(plot_filename, dpi=plot_settings['dpi'])
    print(f"INFO: Final plot saved to {plot_filename}")

    print("\nSUCCESS: Main loop and finalization complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
#     args = parser.parse_args()
#     code = args.code

#     config = load_and_prepare_config('config.yaml')
#     with open('plot_config.yaml', 'r') as f:
#         plot_settings = yaml.safe_load(f)
        

#     apply_global_plot_settings(plot_settings)

#     wavepath = os.path.join(config['waveformpath'], config['datapath'])
#     wavename = os.path.join(
#         wavepath, f'{code}_waveform_cat.h5'
#     )
#     outpath = os.path.join(config['outputpath'], config['datapath'])
#     os.makedirs(outpath, exist_ok=True)

#     loaded_cat = gwg.utils.load_h5(wavename, key="cat")
#     loaded_tdi = gwg.utils.load_h5(wavename, key="tdi")

#     duration = Constants.yr * float(config['duration'])
#     dt = float(config['dt'])
#     df = 1/duration  # The frequency resolution

#     # Define the frequency vector depending on the duration (not really needed here,
#     # but necessary for the definition for the LISA noise function)
#     ndata = int(duration/dt)
#     F     = df*int((ndata+1)/2)
#     fvec  = np.arange(0, F, df) # make the positive frequency vector

#     return_type = 'PSD'
#     sens_kwargs = dict(
#         stochastic_params=None,
#         model=lisa_models.scirdv1,
#         return_type=return_type
#     )
#     sens_mat = AET1SensitivityMatrix(fvec[1:], **sens_kwargs)
#     lisa_noise = {}
#     labels = ['A', 'E', 'T']
#     lisa_noise['f'] = fvec[1:]
#     for k,label in enumerate(labels):
#         lisa_noise[label] = sens_mat[k]

#     icloop_kwargs = config['icloop_kwargs'].copy()
#     icloop_kwargs['snr_thresh2'] = float(icloop_kwargs.pop('snr_cutoff', 7))**2
#     icloop_kwargs['deltaf'] = int(icloop_kwargs.pop('window_size', 1000))

#     # One can split the cat in several pieces to parallelize this call
#     orbits = lisa_models.EqualArmlengthOrbits(use_gpu=config['use_gpu'])
#     GB = GBGPU(orbits=orbits, use_gpu=config['use_gpu'])
#     AET, S1, S1r, cat = gwg.icloop(
#         loaded_tdi, GB, loaded_cat, lisa_noise, **icloop_kwargs
#     )

#     gwg.utils.to_h5(
#         os.path.join(outpath, f'{code}_output_cat.h5'), 
#         cat=cat, tdi=AET, S=S1
#     )

#     # Move any generated PDF files to the output directory, overwriting if they exist
#     # for pdf_file in glob.glob('iter*.pdf'):
#     #     destination_file = os.path.join(outpath, os.path.basename(pdf_file))
#     #     shutil.move(pdf_file, destination_file)

#     fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))

#     A = gwg.utils.FrequencySeries(S1['A'], df=df, kmin=0)
#     E = gwg.utils.FrequencySeries(S1['E'], df=df, kmin=0)
#     T = gwg.utils.FrequencySeries(S1['T'], df=df, kmin=0)
#     f = A.f # Get the frequency array

#     ax.loglog(f, np.absolute(A), label='TDI A', color='green', lw=4, linestyle='--')
#     ax.loglog(f, np.absolute(E), label='TDI E', color='blue', lw=4, linestyle='--')
#     ax.loglog(f, np.absolute(T), label='TDI T', color='red', lw=4, linestyle='--')
#     ax.loglog(lisa_noise["f"], lisa_noise["A"], 'k', lw=1, label='A, E noise')
#     ax.loglog(lisa_noise["f"], lisa_noise["T"], 'dimgrey', lw=1, label='T noise')
#     ax.legend(loc='upper left')
#     ax.set_xlim(1e-4, 1e-1)
#     #ax.set_ylim(1e-47, 1e-36)
#     ax.set_ylabel(r'[$1/\mathrm{Hz}$]')
#     ax.set_xlabel(r'Frequency [$\mathrm{Hz}$]')

#     ax.grid(True,linestyle=':',linewidth='1.')
#     ax.xaxis.set_ticks_position('both')
#     ax.yaxis.set_ticks_position('both')
#     ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

#     fig.tight_layout()
#     fig.savefig(
#         os.path.join(outpath, f'{code}_tdi_noise.png'), 
#         dpi=plot_settings['dpi']
#     )


    
