#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import scipy as sp
import random
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline
from astropy import constants as const
from astropy import units as u
import legwork.source as source
from scipy.optimize import root
import sys
import os
CodeDir       = os.path.dirname(os.path.abspath(__file__))
sys.path[1:1] = [os.path.join(CodeDir, 'PyModules'), os.path.join(CodeDir, 'Data'), os.path.join(CodeDir, 'Simulations')]
from BesanconModelInitParams import BesanconParamsDefined, Alpha, Beta, Gamma
from GalaxyParameters import GalaxyParams
from utils import get_mass_norm
from rapid_code_load_T0 import load_T0_data

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER

# How to run:
# 0) Obtain LISA Synthetic UCB Catalog-related packages: rapid_code_load_T0, get_mass_norm (https://github.com/Synthetic-UCB-Catalogs/analysis-scripts)
# 1) If this is the first time you run the code, set 'RecalculateNormConstants' and 'RecalculateCDFs' to True. For subsequent runs they should be False.
# 2) Ensure that locally there is a './Simulations/' folder with T0-formatted data, with the same stucture as on Google drive; make sure that the ./GalCache/ directory exists
# 3) Ensure there are mass norms defined for the population (i.e. what total Galactic mass naturally contains the current population) from get_mass_norms
# 3) Run the code with 'ImportSimulation' set to True
# 4) Post-process the simulation with the companion visualisation code (Plot3DGal.py)

#Model parameters and options 
ModelParams = { #Main options
               'GalaxyModel': 'Besancon', #Currently can only be Besancon
               'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'RecalculateCDFs': True, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)               
               #Simulation options
               'RunWave': 'initial_condition_variations',
               #'RunSubType': 'fiducial',
               #'RunSubType': 'thermal_ecc',
               #'RunSubType': 'uniform_ecc',
               #'RunSubType': 'm2_min_05',
               #'RunSubType': 'qmin_01',
               'RunSubType': 'porb_log_uniform',
               'Code': 'COSMIC',
               #'Code': 'METISSE',
               #'Code': 'SeBa',     
               #'Code': 'SEVN',
               #'Code': 'ComBinE',
               #'Code': 'COMPAS',
               #Simulation parameters
               'ACutRSunPre': 6., #Initial cut for all DWD binaries
               'UseRepresentingWDs': False, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, N, binaries
               'RepresentDWDsBy': 500000,  #Represent the present-day LISA candidates by this nubmer of binaries
               'LISAPCutHours': (2/1.e-4)/(3600.), #LISA cut-off orbital period, 1.e-4 Hz + remember that GW frequency is 2X the orbital frequency
               'MaxTDelay': 14000,
               'DeltaTGalMyr': 50, #Time step resolution in the Galactic SFR
               #Extra options
               'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualizations
               'OneBinToUse': 10, #Number of the bin, if only one bin in used
               'NPoints': 1e5 # Number of stars to sample if we just sample present-day stars
    }






#Import simulation files

#def print_structure(name, obj):
#    indent = '  ' * (name.count('/') - 1)
#    if isinstance(obj, h5py.Dataset):
#        print(f"{indent}- Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
#    elif isinstance(obj, h5py.Group):
#        print(f"{indent}- Group: {name}")





if ModelParams['ImportSimulation']:
    #Import data
    RunWave         = ModelParams['RunWave']
    RunSubType      = ModelParams['RunSubType']
    Code            = ModelParams['Code']
    if not Code == 'SEVN':
        FileName        = 'Simulations/data_products/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_T0.hdf5'
    else:
        FileName        = 'Simulations/data_products/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_MIST_T0.csv'
    CurrOutDir      = 'Simulations/data_products/simulated_galaxy_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/'
    os.makedirs(CurrOutDir,exist_ok=True)
    ACutRSunPre     = ModelParams['ACutRSunPre']
    LISAPCutHours   = ModelParams['LISAPCutHours'] 
    MaxTDelay       = ModelParams['MaxTDelay']   
    RepresentDWDsBy = ModelParams['RepresentDWDsBy'] 
    DeltaTGalMyr    = ModelParams['DeltaTGalMyr']
    UseRepresentingWDs = ModelParams['UseRepresentingWDs']
    
    #General quantities
    MassNorm        = get_mass_norm(RunSubType)
    NStarsPerRun    = GalaxyParams['MGal']/MassNorm
    if not (Code == 'ComBinE'):
        SimData         = load_T0_data(FileName,Code,metallicity=0.0142)
    else:
        SimData         = load_T0_data(FileName,metallicity=0.0142)
    NRuns           = SimData[1]['NSYS'][0]
    
    #Pre-process simulations
    Sims                          = SimData[0]
    if not (Code == 'ComBinE'):
        DWDSetPre                     = Sims.loc[(Sims.type1.isin([21,22,23])) & (Sims.type2.isin([21,22,23])) & (Sims.semiMajor > 0) & (Sims.semiMajor < ACutRSunPre)].groupby('ID', as_index=False).first() #DWD binaries at the moment of formation with a<6RSun
    else:
        DWDSetPre                     = Sims.loc[(Sims.type1 == 2) & (Sims.type2 == 2) & (Sims.semiMajor > 0) & (Sims.semiMajor < ACutRSunPre)].groupby('ID', as_index=False).first() #DWD binaries at the moment of formation with a<8RSun
    #General properties
    #Lower-mass WD radius
    DWDSetPre['RDonorRSun']       = RWD(np.minimum(DWDSetPre['mass1'],DWDSetPre['mass2']))
    #Mass ratio (lower-mass WD mass/higher-mass WD mass)
    DWDSetPre['qSet']             = np.minimum(DWDSetPre['mass1'],DWDSetPre['mass2'])/np.maximum(DWDSetPre['mass1'],DWDSetPre['mass2'])
    #RLO separation for the lower-mass WD
    DWDSetPre['aRLORSun']         = DWDSetPre['RDonorRSun']/fRL(DWDSetPre['qSet'])
    #Period at DWD formation
    DWDSetPre['PSetDWDFormHours'] = POrbYr(DWDSetPre['mass1'],DWDSetPre['mass2'], DWDSetPre['semiMajor'])*YearToSec/(3600.)  
    #Period at RLO
    DWDSetPre['PSetRLOHours']     = POrbYr(DWDSetPre['mass1'],DWDSetPre['mass2'], DWDSetPre['aRLORSun'])*YearToSec/(3600.) 
    
    #GW-related timescales
    #Point mass GW inspiral time from DWD formation to zero separation
    DWDSetPre['TGWMyrSetTot']            = TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['semiMajor'])
    #Point mass GW inspiral time from DWD formation to LISA band (or zero, if we are in the band already)
    DWDSetPre['aLISABandRSun']           = ABinRSun(DWDSetPre['mass1'], DWDSetPre['mass2'], (LISAPCutHours*3600)/YearToSec)
    DWDSetPre['TGWMyrToLISABandSet']     = (DWDSetPre['TGWMyrSetTot'] - TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['aLISABandRSun'])).clip(0)
    #Point mass GW inspiral time from LISA band (or current location if we are in the band) to RLO    
    DWDSetPre['TGWMyrLISABandToRLOSet']  = TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],np.minimum(DWDSetPre['aLISABandRSun'],DWDSetPre['semiMajor'])) - TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['aRLORSun'])
    #Time from DMS formation to the DWD entering the LISA band
    DWDSetPre['AbsTimeToLISAMyr']        = DWDSetPre['time'] + DWDSetPre['TGWMyrToLISABandSet']
    #Time from DMS formation to the DWD RLO
    DWDSetPre['AbsTimeToLISAEndMyr']     = DWDSetPre['AbsTimeToLISAMyr'] + DWDSetPre['TGWMyrLISABandToRLOSet']
    
    #Select DWDs that: 1)Do not merge upon formation, 2) Will reach the LISA band within the age of the Universe
    DWDSet                       = DWDSetPre.loc[(DWDSetPre.semiMajor > DWDSetPre.aRLORSun) & (DWDSetPre.AbsTimeToLISAMyr < MaxTDelay)].sort_values('AbsTimeToLISAMyr')
    #Total number of DWDs produced in the simulation
    NDWDLISAAllTimesCode         = len(DWDSet.index)
    #Corresponding total number of DWDs ever formed in the MW
    NDWDLISAAllTimesReal         = NDWDLISAAllTimesCode*NStarsPerRun
    
    
    #Get the number of present-day potential LISA sources
    #Track the considered sub-bin
    SubBinCounter    = 0
    #Make a DF that tracks DWD counts, times etc in each sub-bin
    SubBinProps      = []
    #Make a dict that keeps DWD ID pointers for each sub-bin
    SubBinDWDIDDict     = {}
    #Go over each Besancon bin
    for iBin in range(len(BesanconParamsDefined['BinName'])):
        print('Step 1: ', iBin, '/',(len(BesanconParamsDefined['BinName'])))
        #Bin start and end times
        TGalBinStart = BesanconParamsDefined['AgeMin'][iBin]
        TGalBinEnd   = BesanconParamsDefined['AgeMax'][iBin]
        #Galactic mass fraction in the bin        
        GalBinProb   = NormConstantsDict['BinMassFractions'][iBin]
        #Number of sub-bins, equally spaced in time; one sub-bin for starburst bins
        NSubBins     = int(np.floor((TGalBinEnd - TGalBinStart)/DeltaTGalMyr) + 1)
        #Time duration of each sub-bin in this bin
        CurrDeltaT   = (TGalBinEnd - TGalBinStart)/NSubBins
        #Galactic mass fraction per sub-bin
        GalSubBinProb = GalBinProb/NSubBins
        #Initialise the start and end time of the current sub-bin
        CurrTMin      = TGalBinStart
        CurrTMax      = TGalBinStart + CurrDeltaT
        #print('Bin:', iBin)        
        #Loop over sub-bins
        for jSubBin in range(NSubBins):
            #Mid-point in time
            CurrTMid            = 0.5*(CurrTMin + CurrTMax)
            #Current LISA sources (formed before today, will leave the band after today)
            LISASourcesCurrDF   = DWDSet[(DWDSet['AbsTimeToLISAMyr'] < CurrTMid) & (DWDSet['AbsTimeToLISAEndMyr'] > CurrTMid)]
            #The expected number of LISA sources, 
            CurrSubBinNDWDsCode = len(LISASourcesCurrDF.index)
            CurrSubBinDWDReal   = GalSubBinProb*NStarsPerRun*CurrSubBinNDWDsCode
            #Log sub-bin properties
            SubBinProps.append({'SubBinAbsID':SubBinCounter, 'SubBinLocalID':jSubBin, 'BinID':iBin, 'SubBinMidAge': CurrTMid, 'SubBinDeltaT': CurrDeltaT, 
                                'SubBinNDWDsCode': CurrSubBinNDWDsCode, 
                                'SubBinNDWDsReal': CurrSubBinDWDReal})
            #Log DWDs
            SubBinDWDIDDict[SubBinCounter] = LISASourcesCurrDF
            #print(CurrTMin, CurrTMax, NLISASourcesCurr)
            SubBinCounter += 1
            CurrTMin      += CurrDeltaT
            CurrTMax      += CurrDeltaT
            #print(NSubBins)

    #Make a DF for the present-day population properties
    SubBinDF = pd.DataFrame(SubBinProps)
    #Export the population properties

    SubBinDF.to_csv(CurrOutDir + Code + '_Galaxy_LISA_Candidates_Bin_Data.csv', index = False)
    
    #Get overall present-day properties
    #Total real number of LISA sources
    NLISACandidatesToday           = np.sum(SubBinDF['SubBinNDWDsReal'])
    #Total number of simulations available to draw from
    NLISACandidatesTodaySimulated  = np.sum(SubBinDF['SubBinNDWDsCode'])
    #Fraction of DWDs that have formed and become present-day LISA sources
    FracLISADWDsfromAllDWDs        = NLISACandidatesToday/NDWDLISAAllTimesReal
    #What fraction of the needed DWDs we have simulated (approximate number)
    FracSimulated                  = NDWDLISAAllTimesCode/NLISACandidatesToday
    
    #Auxiliary function to make rounding statistically equal to averaged    
    def probabilistic_round(N):
        lower = int(N)
        upper = lower + 1
        fractional_part = N - lower
        return upper if random.random() < fractional_part else lower    
    
        
    #Make a dataset of the present-day LISA DWD candidates
    #Draw the number of objects from each sub-bin in proportion to the number of real DWD LISA candidates expected from this sub-bin
    if UseRepresentingWDs:
        NFindPre     = RepresentDWDsBy
    else:
        NFindPre     = NLISACandidatesToday
    NFindSubBins = np.array([probabilistic_round((NFindPre/NLISACandidatesToday)*SubBinDF['SubBinNDWDsReal'].iloc[i]) for i in range(SubBinCounter)],dtype=int)
    NFind        = np.sum(NFindSubBins)
    PresentDayDWDCandFinSet    = []
    #Do the actual drawing
    for iSubBin in range(SubBinCounter):
        CurrFind     = NFindSubBins[iSubBin]
        if CurrFind > 0:
            PresentDayDWDCandFin   = SubBinDWDIDDict[iSubBin].sample(n=CurrFind, replace=True)
            SubBinRow              = SubBinDF.iloc[iSubBin]
            SubBinData             = pd.DataFrame([SubBinRow.values] * len(PresentDayDWDCandFin), columns=SubBinRow.index)
            PresentDayDWDCandFinSet.append(pd.concat([PresentDayDWDCandFin.reset_index(drop=True), SubBinData.reset_index(drop=True)], axis=1))
                    
    PresentDayDWDCandFinDF = pd.concat(PresentDayDWDCandFinSet, ignore_index=True)
    
    #Find present-day periods:
    PresentDayDWDCandFinDF['ATodayRSun']     = APostGWRSun(PresentDayDWDCandFinDF['mass1'], PresentDayDWDCandFinDF['mass2'], PresentDayDWDCandFinDF['semiMajor'], PresentDayDWDCandFinDF['SubBinMidAge'] - PresentDayDWDCandFinDF['time'])
    PresentDayDWDCandFinDF['PSetTodayHours'] = POrbYr(PresentDayDWDCandFinDF['mass1'],PresentDayDWDCandFinDF['mass2'], PresentDayDWDCandFinDF['ATodayRSun'])*YearToSec/(3600.)
            
else:
    CurrOutDir      = './FieldMSTests/'
    os.makedirs(CurrOutDir,exist_ok=True)
    
######################################################
############ Galaxy Sampling Part
####    

#ModelCache     = PreCompute(ModelParams['OneBinToUse'],'Besancon')

#Routine to load data from an 1-D organised hdf5 file
def load_Rdicts_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as hdf5_file:
        group_names = sorted(hdf5_file.keys(), key=lambda x: int(x.split('_')[1]))
        for group_name in group_names:
            group = hdf5_file[group_name]
            data_dict = {dataset_name: group[dataset_name][:] for dataset_name in group}
            yield data_dict
            
#Routine to load data from a 2D-organised hdf5 file
def load_RZdicts_from_hdf5(file_path):
    ZCDFDictSet = {}
    
    # Open the file for reading
    with h5py.File(file_path, 'r') as hdf5_file:
        # Iterate over each bin group
        for binID in hdf5_file.keys():
            IDString  = int(binID[4:])
            bin_group = hdf5_file[binID]
            
            # Initialize a dictionary to hold the data for this bin
            ZCDFDictSet[IDString] = {}
            
            # Each bin group contains 'r_###' subgroups
            for RID in bin_group.keys():
                r_group   = bin_group[RID]
                RIDString = int(RID[2:])
                
                # Initialize a dict for the data under this r-group
                data_dict = {}
                
                # Each r group has multiple datasets (originally keys in the data_dict)
                for dataset_key in r_group.keys():
                    # Read dataset into memory
                    data_dict[dataset_key] = r_group[dataset_key][...]  # "..." reads the entire dataset
                
                # Store this reconstructed dictionary
                ZCDFDictSet[IDString][RIDString] = data_dict
    return ZCDFDictSet


#Get the R-CDFs
if ModelParams['RecalculateCDFs']: 
    
    #Recalculate the r CDFs first:
    ModelRCache     = []
    for i in range(10):
        ModelRCache.append(PreCompute(i+1,'Besancon'))

    # Create an HDF5 file
    with h5py.File('./GalCache/BesanconRData.h5', 'w') as hdf5_file:
        print('Caching R')
        for idx, data_dict in enumerate(ModelRCache):
            # Create a group for each dictionary
            group = hdf5_file.create_group(f'Rdict_{idx}')
            # Store each list as a dataset within the group
            for key, value in data_dict.items():
                group.create_dataset(key, data=value, compression='gzip')
                
    #Recalculate the z-CDFs:
    
    #Sampling points dimension 1
    iBinSampleSet = [i for i in range(10)]

    # Create another HDF5 file
    with h5py.File('./GalCache/BesanconRZData.h5', 'w') as hdf5_file:
        for iBin in iBinSampleSet:
            print('Caching Bin ' + str(iBin+1))
            # Create a group for each x value
            x_group = hdf5_file.create_group(f'bin_{iBin+1}')
            rSet    = ModelRCache[iBin]['MidRSet']
            rIDs    = list(range(len(rSet)))
            for rID in rIDs:
                if (rID % 100) == 0:
                    print('rID '+ str(rID))
                # Create a subgroup for each y value within the x group
                y_group = x_group.create_group(f'r_{rID}')
                # Compute the function output
                data_dict = GetZCDF(rSet[rID], iBin + 1,'Besancon')
                # Store each list in the dictionary as a dataset
                for key, value in data_dict.items():
                    y_group.create_dataset(key, data=value, compression='gzip')
                    
#Load the previously calculated r CDFs
ModelRCache     = []
for Dict in load_Rdicts_from_hdf5('./GalCache/BesanconRData.h5'):
    # Process each dictionary one at a time
    ModelRCache.append(Dict)        
#Load the previously calculated rz CDFs
ZCDFDictSet = load_RZdicts_from_hdf5('./GalCache/BesanconRZData.h5')


#Get the z-CDFs

def DrawRZ(iBin,Model):
    MidRSet    = ModelRCache[iBin-1]['MidRSet']
    RCDFSet    = ModelRCache[iBin-1]['RCDFSet']
    
    Xir        = np.random.rand()
    RFin       = np.interp(Xir,RCDFSet,MidRSet)
    zFin       = GetZ(RFin,iBin-1,Model)
    
    return [RFin,zFin]

    # def RCDFInv(Xir,Hr):    
    #     # Get the parameters for the inverse CDF
    #     def RCD(R):
    #         Res = (1-np.exp(-R/Hr))-(R/Hr)*np.exp(-R/Hr)-Xir
    #         return Res
        
    #     Sol  = sp.optimize.root_scalar(RCD,bracket=(0.0001*Hr,20*Hr))
    #     if Sol.converged:
    #         R      = Sol.root
    #     else:
    #         print('The radial solution did not converge')
    #         sys.exit()
    #     return R
    
def DrawStar(Model,iBin):
    if iBin == -1:
        BinSet = list(range(1,11))
        iBin   = np.random.choice(BinSet, p=NormConstantsDict['BinMassFractions'])
    RZ     = DrawRZ(iBin,Model)
    Age    = np.random.uniform(BesanconParamsDefined['AgeMin'][iBin-1],BesanconParamsDefined['AgeMax'][iBin-1])
    FeH    = np.random.normal(BesanconParamsDefined['FeHMean'][iBin-1],BesanconParamsDefined['FeHStD'][iBin-1])
    
    Res = {'RZ': RZ, 'Bin': iBin, 'Age': Age, 'FeH': FeH}

    return Res



#Draw the Galactic positions

if ModelParams['ImportSimulation']:
    NGalDo = NFind
else:
    NGalDo = int(ModelParams['NPoints'])
    
RSetFin  = np.zeros(NGalDo)
ZSetFin  = np.zeros(NGalDo)
ThSetFin = np.zeros(NGalDo)
XSetFin  = np.zeros(NGalDo)
YSetFin  = np.zeros(NGalDo)
AgeFin   = np.zeros(NGalDo)
BinFin   = np.zeros(NGalDo)
FeHFin   = np.zeros(NGalDo)


for i in range(NGalDo):
    if i % 10000 == 0:
        print('Step 2: ', i, '/',NGalDo)
    if ModelParams['ImportSimulation']:
        ResCurr     = DrawStar('Besancon', int(PresentDayDWDCandFinDF.iloc[i]['BinID']) + 1)
        AgeFin[i]   = PresentDayDWDCandFinDF.iloc[i]['SubBinMidAge']
    else:
        ResCurr     = DrawStar('Besancon', -1)
        AgeFin[i]   = ResCurr['Age']

    BinFin[i]   = ResCurr['Bin']
    FeHFin[i]   = ResCurr['FeH']
    if not (ResCurr['Bin'] == 10):
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
        
#Remember the right-handedness
XRel     = XSetFin - GalaxyParams['RGalSun']
YRel     = YSetFin
ZRel     = ZSetFin + GalaxyParams['ZGalSun']

RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)
Galb     = np.arcsin(ZRel/RRel)
Gall     = np.zeros(NGalDo)
Gall[YRel>=0] = np.arccos(XRel[YRel>=0]/(np.sqrt((RRel[YRel>=0])**2 - (ZRel[YRel>=0])**2)))
Gall[YRel<0]  = 2*np.pi - np.arccos(XRel[YRel<0]/(np.sqrt((RRel[YRel<0])**2 - (ZRel[YRel<0])**2)))

ResDict  = {'Bin': BinFin, 'Age': AgeFin, 'FeH': FeHFin, 'Xkpc': XSetFin, 'Ykpc': YSetFin, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel, 'Galb': Galb, 'Gall': Gall}
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


    
    
    
    