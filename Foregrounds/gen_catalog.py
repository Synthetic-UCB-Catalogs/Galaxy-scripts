"""
Generates a gwg-compatible catalog in HDF5 format from galaxy simulation catalogs.

This script reads data from a simulation catalog, processes it to compute gravitational wave
parameters, and saves the resulting source catalog in a format suitable for use with the `gwg`
package. The output is an HDF5 file containing the source parameters.

Usage:
    python gen_catalog.py --code <simulation_code>

Example:
    python gen_catalog.py --code COSMIC
"""
import numpy as np
import pandas as pd

import yaml
import os

from astropy.coordinates import SkyCoord, Galactocentric, BarycentricMeanEcliptic
from astropy import units

import argparse
from helpers import Constants, explore_csv

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

seed = None
if config['seed'] is not None:
    seed = config['seed']
rng = np.random.default_rng(seed=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, required=True, help='Name of the code to use.')
    args = parser.parse_args()
    code = args.code
    df = explore_csv(code, config)

    PP = df.PSetTodayHours.values.copy()
    PP *= Constants.hr   # s
    ff = 2/PP  # Hz

    m1 = df.mass1.values.copy()
    m2 = df.mass2.values.copy()

    m1 *= Constants.Msun
    m2 *= Constants.Msun

    eta = m1*m2/(m1+m2)**2
    M = m1 + m2
    Mchirp = M*eta**(3./5)

    tc = 5.*Mchirp/256 * (np.pi*Mchirp*ff)**(-8./3)   # s
    fdot = 3./8 * (ff/tc)   # Hz^2

    RRel = df.RRelkpc.values.copy()  # kpc
    RRel *= Constants.pc*1e+3   # s

    A = Mchirp**(5./3) * ff**(2./3) / RRel

    size = A.size
    incl = np.arccos(-1 + 2*rng.random(size=size))
    pol = 2*np.pi*rng.random(size=size)
    phase = 2*np.pi*rng.random(size=size)

    XX = df.Xkpc.values.copy()  # kpc
    YY = df.Ykpc.values.copy()  # kpc
    ZZ = df.Zkpc.values.copy()  # kpc

    coords = SkyCoord(
        XX, YY, ZZ, 
        unit=(units.kpc,units.kpc,units.kpc), 
        frame=Galactocentric
    )

    Long = coords.transform_to(BarycentricMeanEcliptic).lon.radian
    Lat = coords.transform_to(BarycentricMeanEcliptic).lat.radian

    dtp = [
        ('Name', '<U24'), 
        ('Amplitude', '<f8'), 
        ('EclipticLatitude', '<f8'), 
        ('EclipticLongitude', '<f8'), 
        ('Frequency', '<f8'), 
        ('FrequencyDerivative', '<f8'), 
        ('Inclination', '<f8'), 
        ('InitialPhase', '<f8'), 
        ('Polarization', '<f8')
    ]

    cat = np.recarray(size, dtype=dtp) # Get the empty recarray first

    cat['Name'] = df.UID.values.copy()
    cat['Amplitude'] = A
    cat['EclipticLatitude'] = Long
    cat['EclipticLongitude'] = Lat
    cat['Frequency'] = ff
    cat['FrequencyDerivative'] = fdot
    cat['Inclination'] = incl
    cat['InitialPhase'] = phase
    cat['Polarization'] = pol

    save_dir = os.path.join(config['inputpath'], config['datapath'])
    os.makedirs(save_dir, exist_ok=True)

    catname = f'{code}_input_cat.h5'
    filepath = os.path.join(save_dir, catname)

    df = pd.DataFrame(cat)
    df.to_hdf(filepath, key='cat', mode='w')     

