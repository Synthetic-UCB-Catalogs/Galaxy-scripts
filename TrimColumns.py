#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  12 14:33 2025

@author: alexey
"""

import numpy as np
import pandas as pd
import h5py
from astropy import constants as const
from astropy import units as u
import legwork.source as source
import sys
import os
CodeDir       = os.path.dirname(os.path.abspath(__file__))
sys.path[1:1] = [os.path.join(CodeDir, 'PyModules'), os.path.join(CodeDir, 'Data'), os.path.join(CodeDir, 'Simulations')]

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER

# How to run:
# 0) This is a script for trimming the columns in the LISA DWD files to make them more compact

#Model parameters and options 
ModelParams = {#Simulation options
               'RunWave': 'initial_condition_variations',
               #'OutputFormat': 'h5',
               'OutputFormat': 'csv',
               'RunSubType': 'fiducial',
               #'RunSubType': 'thermal_ecc',
               #'RunSubType': 'uniform_ecc',
               #'RunSubType': 'm2_min_05',
               #'RunSubType': 'qmin_01',
               #'RunSubType': 'porb_log_uniform',
               'Code': 'COSMIC',
               #'Code': 'METISSE',
               #'Code': 'SeBa',     
               #'Code': 'SEVN',
               #'Code': 'ComBinE',
               #'Code': 'COMPAS',
    }

#Import data
RunWave         = ModelParams['RunWave']
RunSubType      = ModelParams['RunSubType']
Code            = ModelParams['Code']
if not Code == 'SEVN':
    FileName        = '../data_products/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_T0.hdf5'
else:
    FileName        = '../data_products/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_MIST_T0.csv'
CurrOutDir      = '../data_products/simulated_galaxy_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/'


InputPath     = CurrOutDir+Code+'_Galaxy_AllDWDs.csv'
OutputPath    = CurrOutDir+Code+'_Galaxy_AllDWDs_Reduced.csv'
HDFOutputPath = CurrOutDir+Code+'_Galaxy_AllDWDs_Reduced.h5'

#Read in the columns
df = pd.read_csv(InputPath, nrows=0)
columns = df.columns.tolist()

ConvertPToFGWQ = True

ColumnsKeep = ['mass1', 'mass2', 'PSetTodayHours', 'RRelkpc', 'Gall', 'Galb']
ColumnNameChange = {'mass1': 'M1MSun', 
                    'mass2': 'M2MSun', 
                    'PSetTodayHours': 'fGWHz', 
                    'RRelkpc': 'dKpc', 
                    'Gall': 'l', 
                    'Galb': 'b'}

chunksize = 5*10**5

#Read only the columns we want, in streaming chunks
reader = pd.read_csv(
    InputPath,
    usecols=ColumnsKeep,
    chunksize=chunksize,
    iterator=True,
)


if ModelParams['OutputFormat'] == 'csv':
    first = True
    for chunk in reader:
        #Rename columns in-place
        if ConvertPToFGWQ:
            chunk['PSetTodayHours'] = (2/(chunk['PSetTodayHours']*60*60))
            chunk.rename(columns=ColumnNameChange, inplace=True)
        else:
            chunk.rename(columns=ColumnNameChange, inplace=True)
        chunk.to_csv(
            OutputPath,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False
elif ModelParams['OutputFormat'] == 'h5':
    first = True
    for chunk in reader:
        #Rename columns in-place
        if ConvertPToFGWQ:
            chunk['PSetTodayHours'] = (2/(chunk['PSetTodayHours']*60*60))
            chunk.rename(columns=ColumnNameChange, inplace=True)
        else:
            chunk.rename(columns=ColumnNameChange, inplace=True)
    
        # write (or append) to HDF5.  Use format='table' so that append works:
        chunk.to_hdf(
            HDFOutputPath,
            key="data",                            # the “group” name inside the file
            mode="w" if first else "a",            # write once, then append
            format="table",                        # appendable format
            index=False,                           # usually you don’t need the index
            append=not first,                      # first write replaces, then we append
            data_columns=True                      # allows querying on columns later
        )
        first = False

    
    