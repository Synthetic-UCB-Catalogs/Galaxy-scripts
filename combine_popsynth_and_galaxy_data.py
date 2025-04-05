#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import legwork as lw
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import MWConsts, fGW_from_A, calculateSeparationAfterSomeTime, inspiral_time
from galaxy_models.draw_samples_from_galaxy import getGalaxySamples
from population_synthesis.get_popsynth_properties import getSimulationProperties

# This environment variable must be set by the user to point to the root of the google drive folder
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']

# Units are MSun, kpc, Gyr
# FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER


def matchDwdsToGalacticPositions(
        pathToPopSynthData=None,
        pathToGalacticSamples=None,
        exportLisaVisibleSources=False, # If true, calculate LISA visibile sources according to Legwork                                                                                                   
    ):

    # Import Population Synthesis data 

    DWDs, Z = getSimulationProperties(pathToPopSynthData)
    m1,   m2,  a_birth, t_DWD = DWDs
    #Msun Msun Rsun     Myr 
    nBinaries = m1.shape[0]

    # Import Galactic position samples
    drawn_samples = getGalaxySamples(pathToGalacticSamples)
    b_gal, l_gal, d_gal, t_birth, which_component = drawn_samples[:,:nBinaries]
    
    # Calculate present day properties of WDs 
    dt = t_birth - t_DWD # Myr

    # WD mass-radius relation
    WD_MR_relation = UnivariateSpline(*np.loadtxt('population_synthesis/WDMassRadiusRelation.dat').T, k=4, s=0)
    r1 = WD_MR_relation(m1) # R_sun 
    r2 = WD_MR_relation(m2) # R_sun

    a_today = calculateSeparationAfterSomeTime(m1, m2, a_birth, dt) # R_sun
    fGW_today = fGW_from_A(m1, m2, a_today) # Hz
    dwd_properties = np.vstack((m1, m2, fGW_today, b_gal, l_gal, d_gal))

    # Remove systems that would have merged by now 
    mask_mergers = a_today < r1+r2
    mask = ~mask_mergers # keep the non-mergers

    # if desired, use legwork to apply additional masking for LISA-visible DWDs
    if exportLisaVisibleSources:

        cutoff = 7 # user should set this

        sources = lw.source.Source(
            m_1   = m1 *u.Msun, 
            m_2   = m2 *u.Msun, 
            dist  = d_gal *u.kpc, 
            f_orb = fGW_today/2 *u.Hz,
            ecc   = np.zeros(nBinaries)
            )

        snr = sources.get_snr(verbose=True)
        mask_detectable_sources = sources.snr > cutoff
        mask &= mask_detectable_sources

    masked_dwds = dwd_properties[:,mask]
    print(masked_dwds.shape)
    return masked_dwds


if __name__ == "__main__":
    matchDwdsToGalacticPositions( pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
                                  pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5")
