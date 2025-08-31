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

import os
import yaml
import argparse
import glob
import shutil

import gwg
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models

from helpers import Constants, apply_global_plot_settings, load_and_prepare_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
    args = parser.parse_args()
    code = args.code

    config = load_and_prepare_config('config.yaml')
    with open('plot_config.yaml', 'r') as f:
        plot_settings = yaml.safe_load(f)
        

    apply_global_plot_settings(plot_settings)

    wavepath = os.path.join(config['waveformpath'], config['datapath'])
    wavename = os.path.join(
        wavepath, f'{code}_waveform_cat.h5'
    )
    outpath = os.path.join(config['outputpath'], config['datapath'])
    os.makedirs(outpath, exist_ok=True)

    loaded_cat = gwg.utils.load_h5(wavename, key="cat")
    loaded_tdi = gwg.utils.load_h5(wavename, key="tdi")

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
    AET, S1, S1r, cat = gwg.icloop(
        loaded_tdi, GB, loaded_cat, lisa_noise, 2000, **config['icloop_kwargs']
    )
    print(cat.size)
    gwg.utils.to_h5(
        os.path.join(outpath, f'{code}_output_cat.h5'), 
        cat=cat
    )

    # Move any generated PDF files to the output directory, overwriting if they exist
    for pdf_file in glob.glob('iter*.pdf'):
        destination_file = os.path.join(outpath, os.path.basename(pdf_file))
        shutil.move(pdf_file, destination_file)

    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))

    A = gwg.utils.FrequencySeries(S1['A'], df=df, kmin=0)
    E = gwg.utils.FrequencySeries(S1['E'], df=df, kmin=0)
    T = gwg.utils.FrequencySeries(S1['T'], df=df, kmin=0)
    f = A.f # Get the frequency array

    ax.loglog(f, np.absolute(A), label='TDI A', color='green', lw=4, linestyle='--')
    ax.loglog(f, np.absolute(E), label='TDI E', color='blue', lw=4, linestyle='--')
    ax.loglog(f, np.absolute(T), label='TDI T', color='red', lw=4, linestyle='--')
    ax.loglog(lisa_noise["f"], lisa_noise["A"], 'k', lw=1, label='A, E noise')
    ax.loglog(lisa_noise["f"], lisa_noise["T"], 'dimgrey', lw=1, label='T noise')
    ax.legend(loc='upper left')
    ax.set_xlim(1e-4, 1e-1)
    #ax.set_ylim(1e-47, 1e-36)
    ax.set_ylabel(r'[$1/\mathrm{Hz}$]')
    ax.set_xlabel(r'Frequency [$\mathrm{Hz}$]')

    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outpath, f'{code}_tdi_noise.png'), 
        dpi=plot_settings['dpi']
    )


    