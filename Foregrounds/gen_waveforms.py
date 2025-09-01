"""
Generates the AET channels of the LISA datastream from a gwg-compatible catalog.

This script uses the `gbgpu` and `lisatools` packages by Michael Katz to generate the
LISA time-domain interferometry (TDI) data streams (A, E, T channels) from a catalog
of gravitational wave sources.

It supports parallel processing on multiple GPUs or sequential processing on a CPU,
controlled by the `use_gpu` and `num_chunks` settings in the config file.

This is the most computationally intensive step in the pipeline. For reference, generating
2 years of data with a 15-second cadence for a catalog of 500,000 sources 
(resulting in ~2 million data points per source per channel) took approximately 4 minutes 
on a 12-core Dell G5 15 laptop with `use_gpu=False`.

Usage:
    python gen_waveforms.py --code <simulation_code>

Example:
    python gen_waveforms.py --code COSMIC
"""
import numpy as np
import pandas as pd
import h5py
import yaml

import os
import sys
import argparse
import glob
import time
import traceback

# Multiprocessing and GPU management
import multiprocessing as mp

import gwg
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models

from helpers import Constants, get_file_hash, load_and_prepare_config, format_bytes


# ==============================================================================
# WORKER FUNCTION
# This function will be executed by each parallel process.
# ==============================================================================
def process_chunk(task_info):
    """
    Worker function to process a single chunk of a binary catalog.
    
    Args:
        task_info (tuple): A tuple containing the necessary information for the task:
            - chunk_idx (int): The index of this chunk (for naming output).
            - catalog_chunk (np.recarray): The subset of the catalog to process.
            - config (dict): The main configuration dictionary.
            - lisa_noise (dict): The pre-calculated LISA noise curves.
            - device_id (int or None): The GPU device ID to use. If None, runs on CPU.
    """
    # Unpack the task information
    chunk_idx, catalog_chunk, config, lisa_noise, device_id = task_info
    
    try:
        # Determine if we are running in GPU mode
        use_gpu = (device_id is not None)

        if use_gpu:
            try:
                import cupy as cp
            except ModuleNotFoundError:
                print("FATAL: Worker process in GPU mode could not find the 'cupy' package.")
                return False
            # CRITICAL: Assign this process to its dedicated GPU.
            cp.cuda.Device(device_id).use()
            print(f"INFO: Processing chunk {chunk_idx + 1}/{config['num_chunks']} on GPU {device_id}...")
        else:
            print(f"INFO: Processing chunk {chunk_idx + 1}/{config['num_chunks']} on CPU...")

        # Initialize all necessary objects within the worker process.
        # This is crucial for both multiprocessing safety and correct GPU context.
        orbits = lisa_models.EqualArmlengthOrbits(use_gpu=use_gpu)
        GB = GBGPU(orbits=orbits, use_gpu=use_gpu)
        
        duration = Constants.yr * float(config['duration'])
        dt = float(config['dt'])

        # --- Perform the core computation ---
        tdi, cat_out = gwg.generate_data(catalog_chunk, lisa_noise, GB, T=duration, dt=dt, AET=True,batch_size=10000,gbgpu_available=True)

        # --- Save the output for this chunk ---
        wavepath = os.path.join(config['waveformpath'], config['datapath'])
        output_filename = os.path.join(wavepath, f"{config['code']}_waveform_cat_{chunk_idx}.h5")
        
        gwg.utils.to_h5(
            output_filename, 
            cat=cat_out, tdi=tdi
        )
        
        print(f"INFO: Chunk {chunk_idx + 1} successfully saved to {output_filename}")
        return True

    except Exception as e:
        # This is vital for debugging. Errors in worker processes can be silent otherwise.
        print(f"!!! ERROR processing chunk {chunk_idx + 1} !!!")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the simulation code to use.')
    args = parser.parse_args()
    code = args.code

    config = load_and_prepare_config('config.yaml')
    config['code'] = code # Add code to config for easy access in worker

    # Add num_chunks to config if not present, default to 1
    if 'num_chunks' not in config:
        config['num_chunks'] = 1
        print("WARNING: 'num_chunks' not found in config. Defaulting to 1.")

    wavepath = os.path.join(config['waveformpath'], config['datapath'])
    os.makedirs(wavepath, exist_ok=True)

    # --- Verification and Cleanup ---
    # We now check against a "finished" flag instead of a single file hash
    finished_flag_path = os.path.join(wavepath, f'{code}_FINISHED.flag')
    current_config_hash = get_file_hash('config.yaml')

    if os.path.exists(finished_flag_path):
        with open(finished_flag_path, 'r') as f:
            previous_config_hash = f.read()
        if previous_config_hash == current_config_hash:
            print("All waveform chunks have already been generated for the current config.")
            sys.exit()

    # If we are proceeding, clean up any old chunk files and the flag
    print("INFO: New or changed configuration detected. Cleaning up old files...")
    for old_file in glob.glob(os.path.join(wavepath, f'{code}_waveform_cat_*.h5')):
        os.remove(old_file)
    if os.path.exists(finished_flag_path):
        os.remove(finished_flag_path)


    # --- Load and Prepare Data ---
    catname = f'{code}_input_cat.h5'
    filepath = os.path.join(config['inputpath'], config['datapath'], catname)
    print(f"INFO: Loading full catalog from {filepath}...")
    full_cat = gwg.utils.load_h5(filepath, key="cat")
    print(f"INFO: Full catalog loaded with {len(full_cat)} sources.")

    # Split the catalog into the specified number of chunks
    num_chunks = config['num_chunks']
    chunks = np.array_split(full_cat, num_chunks)
    print(f"INFO: Catalog split into {num_chunks} chunks.")

    # --- Pre-calculate LISA noise once ---
    duration = Constants.yr * float(config['duration'])
    dt = float(config['dt'])
    df = 1/duration
    ndata = int(duration/dt)
    F = df * int((ndata+1)/2)
    fvec = np.arange(0, F, df)

    sens_kwargs = dict(model=lisa_models.scirdv1, return_type='PSD')
    sens_mat = AET1SensitivityMatrix(fvec[1:], **sens_kwargs)
    lisa_noise = {'f': fvec[1:], 'A': sens_mat[0], 'E': sens_mat[1], 'T': sens_mat[2]}

    # --- BEGIN of Memory Estimation ---
    print("\n--- Memory Estimation ---")
    chunk_memory_sizes = [chunk.nbytes for chunk in chunks]
    for i, mem_size in enumerate(chunk_memory_sizes):
        print(f"  Chunk {i}: {len(chunks[i])} sources, Estimated Memory: {format_bytes(mem_size)}")
    if config['use_gpu']:
        try:
            # You need to import cupy here to check memory
            import cupy as cp
            num_gpus = cp.cuda.runtime.getDeviceCount()
            print("\n--- GPU Memory Check ---")
            for i in range(num_gpus):
                with cp.cuda.Device(i):
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    print(f"  GPU {i}: {format_bytes(free_mem)} free / {format_bytes(total_mem)} total")
            
            # Compare the largest chunk to the free memory on the first GPU
            # (assuming memory is similar across GPUs)
            largest_chunk_size = max(chunk_memory_sizes)
            free_mem_gpu0, _ = cp.cuda.runtime.memGetInfo()
            if largest_chunk_size > free_mem_gpu0:
                print(f"\nWARNING: The largest data chunk ({format_bytes(largest_chunk_size)}) may not fit into the available GPU memory ({format_bytes(free_mem_gpu0)}).")
            else:
                print("\nINFO: Largest data chunk appears to fit into available GPU memory.")

        except (ModuleNotFoundError, RuntimeError) as e:
            print(f"WARNING: Could not perform GPU memory check. Error: {e}")
    
    print("-" * 25)
    # --- END of Memory Estimation ---

    start_time = time.time()

    # --- Execute based on CPU or GPU configuration ---
    if config['use_gpu']:
        # GPU PARALLEL LOGIC
        try:
            import cupy as cp
        except ModuleNotFoundError:
            print("FATAL: 'use_gpu' is True in config, but the 'cupy' package is not installed.")
            sys.exit(1)
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus == 0: raise RuntimeError("No GPUs found.")
            print(f"INFO: Found {num_gpus} GPUs. Starting parallel processing...")
        except Exception as e:
            print(f"FATAL: Could not get GPU count. Exiting. Error: {e}")
            sys.exit(1)
        
        # Create a list of tasks. The GPU device ID is assigned cyclically.
        tasks = [(i, chunks[i], config, lisa_noise, i % num_gpus) for i in range(num_chunks)]
        
        # Create a pool of worker processes, one for each GPU
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(process_chunk, tasks)

    else:
        # CPU SEQUENTIAL LOGIC
        print("INFO: Starting sequential processing on CPU...")
        results = []
        for i in range(num_chunks):
            task_info = (i, chunks[i], config, lisa_noise, None) # None for device_id
            results.append(process_chunk(task_info))

    end_time = time.time()
    print(f"\nINFO: Processing finished in {end_time - start_time:.2f} seconds.")

    # --- Final Verification ---
    if all(results):
        print("INFO: All chunks processed successfully. Writing finished flag.")
        with open(finished_flag_path, 'w') as f:
            f.write(current_config_hash)
    else:
        print("ERROR: One or more chunks failed to process. Please check the logs.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
#     args = parser.parse_args()
#     code = args.code

#     config = load_and_prepare_config('config.yaml')

#     wavepath = os.path.join(config['waveformpath'], config['datapath'])
#     os.makedirs(wavepath, exist_ok=True)

#     # --- Verification Logic ---
#     waveform_filepath = os.path.join(wavepath, f'{code}_waveform_cat.h5')
#     config_hash_path = os.path.join(wavepath, f'{code}_config.sha256')

#     current_config_hash = get_file_hash('config.yaml')

#     if os.path.exists(waveform_filepath):
#         if os.path.exists(config_hash_path):
#             with open(config_hash_path, 'r') as f:
#                 previous_config_hash = f.read()
#             if previous_config_hash == current_config_hash:
#                 print(f"Waveform file '{waveform_filepath}' has already been generated for the current config.")
#                 sys.exit()

#     catname = f'{code}_input_cat.h5'
#     filepath = os.path.join(
#         config['inputpath'], config['datapath'], catname
#     )
#     cat = gwg.utils.load_h5(filepath, key="cat")

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


#     # One can split the cat in several pieces to parallelize this call
#     orbits = lisa_models.EqualArmlengthOrbits(use_gpu=config['use_gpu'])
#     GB = GBGPU(orbits=orbits, use_gpu=config['use_gpu'])
#     tdi, cat = gwg.generate_data(cat, lisa_noise, GB, T=duration, dt=dt, AET=True)

#     # Save the data
#     gwg.utils.to_h5(
#         waveform_filepath, 
#         cat=cat, tdi=tdi
#     )

#     # Save the config hash for future verification
#     with open(config_hash_path, 'w') as f:
#         f.write(current_config_hash)


    
