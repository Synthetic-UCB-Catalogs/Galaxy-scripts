"""
Generates the AET channels of the LISA datastream from a gwg-compatible catalog.

This script uses the `gbgpu` and `lisatools` packages by Michael Katz to generate the
LISA time-domain interferometry (TDI) data streams (A, E, T channels) from a catalog
of gravitational wave sources.

This is the most computationally intensive step in the pipeline. For reference, generating
2 years of data with a 15-second cadence (resulting in ~2 million data points) took
approximately 4 minutes on a 12-core Dell G5 15 laptop with `use_gpu=False`.

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
import argparse
import sys

import gwg
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models

from helpers import Constants, get_file_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
    args = parser.parse_args()
    code = args.code

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    wavepath = os.path.join(config['waveformpath'], config['datapath'])
    os.makedirs(wavepath, exist_ok=True)

    # --- Verification Logic ---
    waveform_filepath = os.path.join(wavepath, f'{code}_waveform_cat.h5')
    config_hash_path = os.path.join(wavepath, f'{code}_config.sha256')

    current_config_hash = get_file_hash('config.yaml')

    if os.path.exists(waveform_filepath):
        if os.path.exists(config_hash_path):
            with open(config_hash_path, 'r') as f:
                previous_config_hash = f.read()
            if previous_config_hash == current_config_hash:
                print(f"Waveform file '{waveform_filepath}' has already been generated for the current config.")
                sys.exit()

    catname = f'{code}_input_cat.h5'
    filepath = os.path.join(
        config['inputpath'], config['datapath'], catname
    )
    cat = gwg.utils.load_h5(filepath, key="cat")

    duration = Constants.yr * float(config['duration'])
    dt = float(config['dt'])
    df = 1/duration  # The frequency resolution

    # Define the frequency vector depending on the duration (not really needed here,
    # but necessary for the definition for the LISA noise function)
    ndata = int(duration/dt)
    F     = df*int((ndata+1)/2)
    fvec  = np.arange(0, F, df) # make the positive frequency vector

    return_type = 'PSD'
    sens_kwargs = dict(
        stochastic_params=None,
        model=lisa_models.scirdv1,
        return_type=return_type
    )
    sens_mat = AET1SensitivityMatrix(fvec[1:], **sens_kwargs)
    lisa_noise = {}
    labels = ['A', 'E', 'T']
    lisa_noise['f'] = fvec[1:]
    for k,label in enumerate(labels):
        lisa_noise[label] = sens_mat[k]


    # One can split the cat in several pieces to parallelize this call
    GB = GBGPU(use_gpu=config['use_gpu'])
    tdi, cat = gwg.generate_data(cat, lisa_noise, GB, T=duration, dt=dt, AET=True)

    # Save the data
    gwg.utils.to_h5(
        waveform_filepath, 
        cat=cat, tdi=tdi
    )

    # Save the config hash for future verification
    with open(config_hash_path, 'w') as f:
        f.write(current_config_hash)


    