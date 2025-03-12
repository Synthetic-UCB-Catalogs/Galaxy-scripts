#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import os, sys
import numpy as np
import pandas as pd
import scipy as sp
import random
import h5py
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.optimize import root
import legwork.source as source
from GalaxyModels.getGalaxyModel import getGalaxyModel
from utils import MWConsts

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER

def generateSimulatedDwdGalaxy(
    galaxyModelName,
    useOneComponentOnly=False, #If False - use full model; if True - use just one bin, for visualizations
    componentToUse=10, #Number of the bin, if only one bin in used
    recalculateNormConstants= True, #False, # If true, density normalisations are recalculated and printed out, else already existing versions are used               
    recalculateCDFs= True, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
    importSimulation= True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)
    nPoints= 1e5, # Number of stars to sample if we just sample present-day stars

    ):

    galaxyModel = getGalaxyModel(galaxyModelName, recalculateNormConstants=recalculateNormConstants, recalculateCDFs=recalculateCDFs)
    galacticComponentMassFractions = galaxyModel.getGalacticComponentMassFractions()
    RZ_CDFs = galaxyModel.getRZ_CDFs()
    
    MRWDMset,MRWDRSet    = np.split(np.loadtxt(CodeDir + '/WDData/MRRel.dat'),2,axis=1)
    MRWDMset             = MRWDMset.flatten()
    MRWDRSet             = MRWDRSet.flatten()
    MRSpl                = UnivariateSpline(MRWDMset, MRWDRSet, k=4, s=0)


    #Import simulation files
    if importSimulation:
        getSimulationProperties()
    else:
        CurrOutDir      = './FieldMSTests/'
        os.makedirs(CurrOutDir,exist_ok=True)

    
    
    
    
    
    
    
        
    
    
    
    #Draw the Galactic positions
    
    if ModelParams['ImportSimulation']:
        NGalDo = NFind
    else:
        NGalDo = nPoints
        
    RSetFin  = np.zeros(NGalDo)
    ZSetFin  = np.zeros(NGalDo)
    ThSetFin = np.zeros(NGalDo)
    XSetFin  = np.zeros(NGalDo)
    YSetFin  = np.zeros(NGalDo)
    AgeFin   = np.zeros(NGalDo)
    ComponentFin   = np.zeros(NGalDo)
    FeHFin   = np.zeros(NGalDo)
    
    
    # RTW fix this - don't do for loop over 1e5 draws!
    for i in range(NGalDo):
        if i % 100 == 0:
            print('Step 2: ', i, '/',NGalDo)
        if ModelParams['ImportSimulation']:
            ResCurr     = DrawStar('Besancon', int(PresentDayDWDCandFinDF.iloc[i]['ComponentID']) + 1)
            AgeFin[i]   = PresentDayDWDCandFinDF.iloc[i]['SubComponentMidAge']
        else:
            ResCurr     = DrawStar('Besancon', -1)
            AgeFin[i]   = ResCurr['Age']
    
        ComponentFin[i]   = ResCurr['Component']
        FeHFin[i]   = ResCurr['FeH']
        if not (ResCurr['Component'] == 10):
            RSetFin[i]  = ResCurr['RZ'][0]
            ZSetFin[i]  = ResCurr['RZ'][1]
            Th          = 2.*np.pi*np.random.uniform()
            ThSetFin[i] = Th
            XSetFin[i]  = ResCurr['RZ'][0]*np.cos(Th)
            YSetFin[i]  = ResCurr['RZ'][0]*np.sin(Th)
        else:
            #The bulge
            #R and Z are such that Z is the -X axis of the bulge, and R=(X,Y) are (Y,-Z) axes of the bulge
            #First transform to bulge coordinates
            Th          = 2.*np.pi*np.random.uniform()
            Rad         = ResCurr['RZ'][0]
            XPrime      = Rad*np.cos(Th)
            YPrime      = Rad*np.sin(Th)
            ZPrime      = ResCurr['RZ'][1]
            #ASSUMING THE ALPHA ANGLE IS ALONG THE GALACTIC ROTATION - CHECK DWEK
            XSetFin[i]  = -ZPrime*np.sin(Alpha) + XPrime*np.cos(Alpha)
            YSetFin[i]  = ZPrime*np.cos(Alpha) + XPrime*np.sin(Alpha)
            ZSetFin[i]  = -YPrime
            RSetFin[i]  = np.sqrt(XPrime**2 + ZPrime**2)
            
    
    # Calculate and convert to different frame
    #Remember the right-handedness
    XRel     = XSetFin - MWConsts['RGalSun']
    YRel     = YSetFin
    ZRel     = ZSetFin + MWConsts['ZGalSun']
    
    RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)
    Galb     = np.arcsin(ZRel/RRel)
    Gall     = np.zeros(NGalDo)
    Gall[YRel>=0] = np.arccos(XRel[YRel>=0]/(np.sqrt((RRel[YRel>=0])**2 - (ZRel[YRel>=0])**2)))
    Gall[YRel<0]  = 2*np.pi - np.arccos(XRel[YRel<0]/(np.sqrt((RRel[YRel<0])**2 - (ZRel[YRel<0])**2)))
    
    ResDict  = {'Component': ComponentFin, 'Age': AgeFin, 'FeH': FeHFin, 'Xkpc': XSetFin, 'Ykpc': YSetFin, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel, 'Galb': Galb, 'Gall': Gall}
    ResDF    = pd.DataFrame(ResDict)
    
    #DWDDF    = DWDSet.iloc[IDSet]
    if ModelParams['ImportSimulation']:
        ResDF      = pd.concat([ResDF, PresentDayDWDCandFinDF], axis=1)
        ResDF.to_csv(CurrOutDir+Code+'_Galaxy_AllDWDs.csv', index = False)
    else:
        ResDF.to_csv(CurrOutDir + '/FullGalaxyMSs.csv', index = False)
    
    #Export only LISA-visible DWDs
    if ModelParams['ImportSimulation']:
        n_values = len(ResDF.index)
    
        m_1    = (ResDF['mass1']).to_numpy() * u.Msun
        m_2    = (ResDF['mass2']).to_numpy() * u.Msun
        dist   = (ResDF['RRelkpc']).to_numpy() * u.kpc
        f_orb = (1/(ResDF['PSetTodayHours']*60*60)).to_numpy() * u.Hz
        ecc   = np.zeros(n_values)
        
        sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb)
        snr     = sources.get_snr(verbose=True)
        
        cutoff = 7
        
        detectable_threshold = cutoff
        detectable_sources   = sources.snr > cutoff
        
        LISADF = ResDF[detectable_sources]
        
    
        LISADF.to_csv(CurrOutDir+Code+'_Galaxy_LISA_DWDs.csv', index = False)
    

    
    
    
    
if __name__ == "__main__":
    generateSimulatedDwdGalaxy(galaxyModelName='Besancon')









    #Model parameters and options
    #ModelParams = {
    #    }
    
    #Galaxy model can be 'Besancon'
    #if ModelParams['GalaxyModel'] == 'Besancon':
    #    GalaxyModelParams = BesanconParams
    #    GalFunctionsDict = {'Besancon': GalaxyModelParams['RhoFunction']}
    
    #Find the norm for the different Galactic components, and define the weights
    #Store it in # NormCSet - the normalisation constant array
    
    #def print_structure(name, obj):
    #    indent = '  ' * (name.count('/') - 1)
    #    if isinstance(obj, h5py.Dataset):
    #        print(f"{indent}- Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    #    elif isinstance(obj, h5py.Group):
    #        print(f"{indent}- Group: {name}")
    
##CodeDir       = os.path.dirname(os.path.abspath(__file__))
##sys.path[1:1] = [os.path.join(CodeDir, 'PyModules'), os.path.join(CodeDir, 'Data'), os.path.join(CodeDir, 'Simulations')]
#from BesanconModel import BesanconParams, Alpha, Beta, Gamma
