#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey, reinhold
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from BesanconModel import BesanconModel
# TODO: implement other models


def getGalaxyModel(galaxyModelName, *args, **kwargs):
    if galaxyModelName == "Besancon":
        return BesanconModel("Besancon", *args, **kwargs)
    else:
        raise Exception("Model not configured")


def generateSimulatedGalaxy(
        galaxyModelName,                 # (str) The name of the Galaxy Model to use
        nSamples=1e6,                    # (int) Number of stars to sample 
        Z=0.0142,                        # (float) Metallicity of samples
        fnameOutput=None,                # (str) File name to save (only relevant if saveOutput is true)
        saveOutput=False,                # (bool) Whether to save the output to a file (if new samples are drawn)
        makePlot=False,                  # (bool) Whether to generate a 3D plot of the data
        singleComponentToUse=None,       # (int) Number of the single component to model (for visualizations). If None, use full model.
    ):

    if not saveOutput:
        fnameOutput=None

    # Create Galaxy model and draw (or import) samples of binary locations and birth times
    galaxyModel = getGalaxyModel(galaxyModelName, Z, fnameOutput, saveOutput, singleComponentToUse) 
    drawn_samples = galaxyModel.GetSamples(int(nSamples))
    b_gal, l_gal, d_gal, t_birth, which_component = drawn_samples

    if makePlot:
        z_gal = d_gal*np.sin(b_gal) 
        x_gal = d_gal*np.cos(b_gal)*np.cos(l_gal)
        y_gal = d_gal*np.cos(b_gal)*np.sin(l_gal)

        cmap = mpl.cm.rainbow
        color = cmap(which_component/10) # color by component
        #color = cmap(np.arange(len(d_gal))/len(d_gal)) # color by position in the samples list - which is ordered by distance from the sun

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_gal, y_gal, z_gal, s=1, c=color) 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set_zlim(-30,30)

        # Plot individual components separately
        #fig, axes = plt.subplots(ncols=5, nrows=2)
        #axs = axes.flatten()
        #for ii in range(10):
        #    mask = which_component == ii
        #    label = ['ThinDisk1',  'ThinDisk2',  'ThinDisk3',  'ThinDisk4',  'ThinDisk5',  'ThinDisk6',  'ThinDisk7',  'ThickDisk',      'Halo',   'Bulge'][ii]
        #    ax = axs[ii]
        #    ax.scatter(r_gal[mask], z_gal[mask], s=1, c=color[mask], label=label)
        #    ax.legend(fontsize=12)
        #    ax.set_xlim(0, 30)
        #    ax.set_ylim(-30,30)
        plt.show()

if __name__ == "__main__":
    generateSimulatedGalaxy(galaxyModelName='Besancon', makePlot=False, saveOutput=True) 
