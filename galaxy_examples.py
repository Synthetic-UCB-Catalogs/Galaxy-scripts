#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 23:56:58 2025

@author: alexey
"""

# +
## A preamble that allows one to access the package files from within the same folder
#if __package__ is None:            # executed only if you pressed "Run file"
#    import pathlib, sys, types
#    pkg_dir  = pathlib.Path(__file__).resolve().parent
#    pkg_name = pkg_dir.name                     # Current folder
#
#    # create (or reuse) a dummy top‑level package object
#    if pkg_name in sys.modules:
#        pkg = sys.modules[pkg_name]
#    else:
#        pkg = types.ModuleType(pkg_name)
#        sys.modules[pkg_name] = pkg
#
#    pkg.__path__ = [str(pkg_dir)]               # this makes it a package
#    sys.path.insert(0, str(pkg_dir.parent))     # absolute imports still work
#    __package__ = pkg_name
## -----------------------------------------------------------------------

# +
# Other packages for the example
import os, sys

# Add local scripts
sys.path.insert(0, '..')
from galaxy_scripts.combine_popsynth_and_galaxy_data import matchStuff # fix this
# -

# This environment variable must be set by the user to point to the root of the google drive folder
# See https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA/edit?usp=sharing for instructions on how to set up the google drive structure locally
ROOT_DIR = os.environ['UCB_ROOT_DIR']
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']

# The actual example script
cp.matchDwdsToGalacticPositions(pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
                             #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
                              useLegworkMask=False, 
                              applyInitialLisaBandFilter=False)
