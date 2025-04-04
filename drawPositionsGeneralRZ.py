#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as ss
import random
import h5py
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.optimize import root
import legwork.source as source
from GalaxyModels.getGalaxyModel import getGalaxyModel
from Simulations.getPopSynthProperties import getSimulationProperties
from utils import MWConsts, P_from_A, calculateSeparationAfterSomeTime, inspiral_time
import matplotlib as mpl
import matplotlib.pyplot as plt

# Units are MSun, kpc, Gyr
# FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER


def generateSimulatedDwdGalaxy(
    galaxyModelName,               
    importSimulation=None,          # If not None, should be a path to a *_T0.h5 popsynth output file - this will construct the present-day DWD populaiton (None will just produce the MS population)
    redrawSamples=False,            # If True (or if file doesn't exist), draw samples from Galaxy model
    exportLisaVisibleSources=False, # If true, calculate LISA visibile sources according to Legwork                                                                                                   
    # Parameters for testing purposes only
    singleComponentToUse=None,      # Number of the single component to model (for visualizations). If None, use full model.
    nBinaries=int(1e6),               # Number of stars to sample if we just sample present-day stars
):

    # Import simulation files
    if importSimulation is not None:
        df, header = getSimulationProperties(importSimulation)
        nBinaries = df.shape[0]
        Z = header['Z'][0]
        m1 = df.mass1           # Msun
        m2 = df.mass2           # Msun
        a_birth = df.semiMajor  # Rsun
        t_DWD = df.time         # Myr - formation time
        testing = False
    else:
        Z = 0.0142
        testing = True


    # Create Galaxy model and draw (or import) samples of binary locations and birth times
    galaxyModel = getGalaxyModel(galaxyModelName) #, recalculateNormConstants=recalculateNormConstants, recalculateCDFs=recalculateCDFs)
    b_gal, l_gal, d_gal, t_birth, which_component = galaxyModel.GetSamples(nBinaries, Z, redrawSamples)

    if testing:
        z_gal = d_gal*np.sin(b_gal) 
        x_gal = d_gal*np.cos(b_gal)*np.cos(l_gal)
        y_gal = d_gal*np.cos(b_gal)*np.sin(l_gal)

        cmap = mpl.cm.rainbow
        color = cmap(which_component/10)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_gal, y_gal, z_gal, s=1, c=color) #, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set_zlim(-30,30)

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
        return

    # Calculate present day properties of WDs 
    dt = t_birth - t_DWD

    # WD mass-radius relation
    WD_MR_relation = UnivariateSpline(*np.loadtxt('WDData/WDMassRadiusRelation.dat').T, k=4, s=0)
    r1 = WD_MR_relation(m1) # R_sun 
    r2 = WD_MR_relation(m2) # R_sun

    P_birth = P_from_A(m1, m2, a_birth)     # yr
    tau_GW = inspiral_time(m1, m2, a_birth) # Myr
    
    a_today = calculateSeparationAfterSomeTime(m1, m2, a_birth, dt) # R_sun
    P_today = P_from_A(m1, m2, a_today)                             # yr


    print(P_today)



    # popsynth says for this amount of mass in SFR for given Z (after correcting), here is a DWD with X properties, after this amount of time...

    # The idea is that you should be weighting your choice of component by the likelihood of that component. Currently we only do based on current mass,
    # but shouldn't we also factor in that a component that's been


    print(done)
    ##

    # Draw the Galactic positions

    #RSetFin = np.zeros(NGalDo)
    #ZSetFin = np.zeros(NGalDo)
    #ThSetFin = np.zeros(NGalDo)
    #XSetFin = np.zeros(NGalDo)
    #YSetFin = np.zeros(NGalDo)
    #AgeFin = np.zeros(NGalDo)
    #ComponentFin = np.zeros(NGalDo)
    #FeHFin = np.zeros(NGalDo)

    # RTW fix this - don't do for loop over 1e5 draws!
    # for i in range(NGalDo):
    #    if i % 100 == 0:
    #        print('Step 2: ', i, '/',NGalDo)
    #    if ModelParams['ImportSimulation']:
    #        ResCurr     = DrawStar('Besancon', int(PresentDayDWDCandFinDF.iloc[i]['ComponentID']) + 1)
    #        AgeFin[i]   = PresentDayDWDCandFinDF.iloc[i]['SubComponentMidAge']
    #    else:
    #        ResCurr     = DrawStar('Besancon', -1)
    #        AgeFin[i]   = ResCurr['Age']
    #
    #    ComponentFin[i]   = ResCurr['Component']
    #    FeHFin[i]   = ResCurr['FeH']
    #    if not (ResCurr['Component'] == 10):
    #        RSetFin[i]  = ResCurr['RZ'][0]
    #        ZSetFin[i]  = ResCurr['RZ'][1]
    #        Th          = 2.*np.pi*np.random.uniform()
    #        ThSetFin[i] = Th
    #        XSetFin[i]  = ResCurr['RZ'][0]*np.cos(Th)
    #        YSetFin[i]  = ResCurr['RZ'][0]*np.sin(Th)
    #    else:
    #        #The bulge
    #        #R and Z are such that Z is the -X axis of the bulge, and R=(X,Y) are (Y,-Z) axes of the bulge
    #        #First transform to bulge coordinates
    #        Th          = 2.*np.pi*np.random.uniform()
    #        Rad         = ResCurr['RZ'][0]
    #        XPrime      = Rad*np.cos(Th)
    #        YPrime      = Rad*np.sin(Th)
    #        ZPrime      = ResCurr['RZ'][1]
    #        #ASSUMING THE ALPHA ANGLE IS ALONG THE GALACTIC ROTATION - CHECK DWEK
    #        XSetFin[i]  = -ZPrime*np.sin(Alpha) + XPrime*np.cos(Alpha)
    #        YSetFin[i]  = ZPrime*np.cos(Alpha) + XPrime*np.sin(Alpha)
    #        ZSetFin[i]  = -YPrime
    #        RSetFin[i]  = np.sqrt(XPrime**2 + ZPrime**2)

    # Calculate and convert to different frame
    # Remember the right-handedness
    #XRel = XSetFin - MWConsts['RGalSun']
    #YRel = YSetFin
    #ZRel = ZSetFin + MWConsts['ZGalSun']

    #RRel = np.sqrt(XRel**2 + YRel**2 + ZRel**2)
    #Galb = np.arcsin(ZRel/RRel)
    #Gall = np.zeros(NGalDo)
    #Gall[YRel >= 0] = np.arccos(
    #    XRel[YRel >= 0]/(np.sqrt((RRel[YRel >= 0])**2 - (ZRel[YRel >= 0])**2)))
    #Gall[YRel < 0] = 2*np.pi - \
    #    np.arccos(XRel[YRel < 0] /
    #              (np.sqrt((RRel[YRel < 0])**2 - (ZRel[YRel < 0])**2)))

    #ResDict = {'Component': ComponentFin, 'Age': AgeFin, 'FeH': FeHFin, 'Xkpc': XSetFin, 'Ykpc': YSetFin, 'Zkpc': ZSetFin,
    #           'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc': YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel, 'Galb': Galb, 'Gall': Gall}
    #ResDF = pd.DataFrame(ResDict)

    ## DWDDF    = DWDSet.iloc[IDSet]
    #if ModelParams['ImportSimulation']:
    #    ResDF = pd.concat([ResDF, PresentDayDWDCandFinDF], axis=1)
    #    ResDF.to_csv(CurrOutDir+Code+'_Galaxy_AllDWDs.csv', index=False)
    #else:
    #    ResDF.to_csv(CurrOutDir + '/FullGalaxyMSs.csv', index=False)

    # Export only LISA-visible DWDs
    if exportLisaVisibleSources:
        n_values = len(ResDF.index)

        m_1 = (ResDF['mass1']).to_numpy() * u.Msun
        m_2 = (ResDF['mass2']).to_numpy() * u.Msun
        dist = (ResDF['RRelkpc']).to_numpy() * u.kpc
        f_orb = (1/(ResDF['PSetTodayHours']*60*60)).to_numpy() * u.Hz
        ecc = np.zeros(n_values)

        sources = source.Source(
            m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb)
        snr = sources.get_snr(verbose=True)

        cutoff = 7

        detectable_threshold = cutoff
        detectable_sources = sources.snr > cutoff

        LISADF = ResDF[detectable_sources]

        LISADF.to_csv(CurrOutDir+Code+'_Galaxy_LISA_DWDs.csv', index=False)


if __name__ == "__main__":
    generateSimulatedDwdGalaxy(galaxyModelName='Besancon', importSimulation="simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5")
    #generateSimulatedDwdGalaxy(galaxyModelName='Besancon') 

