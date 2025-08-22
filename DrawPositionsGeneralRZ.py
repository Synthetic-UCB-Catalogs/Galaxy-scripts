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
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy import constants as const
from astropy import units as u
import legwork.source as source
from scipy.optimize import root
import time
import sys
import os
CodeDir       = os.path.dirname(os.path.abspath(__file__))
sys.path[1:1] = [os.path.join(CodeDir, 'PyModules'), os.path.join(CodeDir, 'Data'), os.path.join(CodeDir, 'Simulations')]
from BesanconModelInitParams import BesanconParamsDefined, Alpha, Beta, Gamma
from GalaxyParameters import GalaxyParams
from get_mass_norm import get_mass_norm
from rapid_code_load_T0 import load_T0_data

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER

# How to run:
# 0) Obtain LISA Synthetic UCB Catalog-related packages: rapid_code_load_T0, get_mass_norm (https://github.com/Synthetic-UCB-Catalogs/analysis-scripts)
# 1) If this is the first time you run the code, set 'RecalculateNormConstants' and 'RecalculateCDFs' to True. For subsequent runs they should be False.
# 2) Ensure that locally there is a './Simulations/' folder with T0-formatted data, with the same stucture as on Google drive
# 3) Ensure there are mass norms defined for the population (i.e. what total Galactic mass naturally contains the current population)
# 3) Run the code with 'ImportSimulation' set to True
# 4) Post-process the simulation with the companion visualisation code (Plot3DGal.py)

#Model parameters and options 
ModelParams = { #Main options
               'GalaxyModel': 'Besancon', #Currently can only be Besancon
               'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'RecalculateCDFs': False, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)               
               #Simulation options
               'RunWave': 'initial_condition_variations',
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

#Time the run
start = time.perf_counter()


######################################
#######    Galactic Model Specifications
###

#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)

#Define the model in two steps:
#First, specify already known parameters

#Rho(r,z), for the Besancon model, thin disk, weights are defined later
def RhoBesancon(r, z, iBin):
    global Alpha, Beta, Gamma
    #print(iBin)
    #Young thin disc
    if iBin == 1:
        hPlus  = 5
        hMinus = 3
        Eps    = BesanconParamsDefined['EpsSetThin'][0]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(aParam**2/hPlus**2)) - np.exp(-(aParam**2/hMinus**2))
    #Thin disk - other bins
    elif (iBin >= 2) and (iBin <=7):
        hPlus  = 2.53
        hMinus = 1.32
        Eps    = BesanconParamsDefined['EpsSetThin'][iBin - 1]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(0.5**2 + aParam**2/hPlus**2)**0.5) - np.exp(-(0.5**2 + aParam**2/hMinus**2)**0.5)
    #Thick disc
    elif (iBin == 8):
        xl   = 0.4
        RSun = 8
        hR   = 2.5
        hz   = 0.8
        Res  = np.where(np.abs(z) <=xl, (np.exp(-(r-RSun)/hR))*(1.-((1/hz)/(xl*(2+xl/hz)))*(z**2)),
                        (np.exp(-(r-RSun)/hR))*((np.exp(xl/hz))/(1+xl/(2*hz)))*(np.exp(-np.abs(z)/hz)))
    #Halo
    elif (iBin == 9):
        ac   = 0.5
        Eps  = 0.76
        RSun = 8. 
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.where((aParam<=ac), (ac/RSun)**(-2.44),(aParam/RSun)**(-2.44))
    #Bulge
    elif (iBin == 10):
        x0 = 1.59
        yz0 = 0.424
        #yz0 = 0.424 -- y0 and z0 are equal, use that to sample the bulge stars in the coordinates where z-axis is aligned with the x-axis of the bugle
        Rc = 2.54
        N  = 13.7
        #See Robin Tab 5
        #Orientation angles: α (angle between the bulge major axis and the line perpendicular to the Sun – Galactic Center line), 
        #β (tilt angle between the bulge plane and the Galactic plane) and 
        #γ (roll angle around the bulge major axis);

        #We assume z is the axis of symmetry, but in the bulge coordinates it is x; use rotation
        xbulge = -z
        #Note, bulge is not fully axisymmetric, and minor axes y and z contribute differently to the equation
        #REVISE AND SAMPLE FROM 3D
        rbulge = r        

        #rs2    = np.sqrt(((x/x0)**2 + (y/yz0)**2)**2 + (z/z0)**4)                
        rs2    = np.sqrt(((rbulge/yz0)**2)**2 + (xbulge/x0)**4)   
        rParam = rbulge
        Res    = np.where(rParam<=Rc, np.exp(-0.5*rs2),
            #np.exp(-0.5*rs2)*np.exp(-0.5*((np.sqrt(x**2 + y**2) - Rc)/(0.5))**2)
            np.exp(-0.5*rs2)*np.exp(-0.5*((rbulge - Rc)/(0.5))**2))
        
    return Res


#3D version for later
# =============================================================================
## Volume integrator for the Galactic density components
# def GetVolumeIntegral(iBin):
#     NPoints =  BesanconParamsDefined['ZNPoints'][iBin-1]
#     
#     XRange = BesanconParamsDefined['XRange'][iBin-1]
#     YRange = BesanconParamsDefined['YRange'][iBin-1]
#     ZRange = BesanconParamsDefined['ZRange'][iBin-1]
#     
#     
#     XSet = np.linspace(-XRange, XRange, NPoints)
#     YSet = np.linspace(-YRange, YRange, NPoints)
#     ZSet = np.linspace(-ZRange, ZRange, NPoints)
#     
#     dX = 2 * XRange / (NPoints - 1)
#     dY = 2 * YRange / (NPoints - 1)
#     dZ = 2 * ZRange / (NPoints - 1)
#  
#     def RhoFun(X, Y, Z):
#         return RhoBesancon(np.sqrt(X**2 + Y**2), Z, iBin)
#     
#     X, Y, Z = np.meshgrid(XSet, YSet, ZSet, indexing='ij')
#     RhoSet = RhoFun(X, Y, Z)
#     
#     Res = np.sum(RhoSet) * dX * dY * dZ
#     
#     return Res
# =============================================================================


#2D Volume integrator for the Galactic density components
def GetVolumeIntegral(iBin):
    NPoints =  BesanconParamsDefined['ZNPoints'][iBin-1]
    
    RRange = np.sqrt((BesanconParamsDefined['XRange'][iBin-1])**2 + (BesanconParamsDefined['YRange'][iBin-1])**2)
    ZRange = BesanconParamsDefined['ZRange'][iBin-1]
    
    
    RSet = np.linspace(0, RRange, NPoints)
    ZSet = np.linspace(-ZRange, ZRange, NPoints)
    
    dR = RRange / (NPoints - 1)
    dZ = 2 * ZRange / (NPoints - 1)
  
    def RhoFun(R, Z):
        return RhoBesancon(R, Z, iBin)
    
    R, Z = np.meshgrid(RSet, ZSet, indexing='ij')
    RhoSet = RhoFun(R, Z)
    
    Res = np.sum(RhoSet*2*np.pi*R) * dR * dZ
    
    return Res

#Find the norm for the different Galactic components, and define the weights
#Store it in # NormCSet - the normalisation constant array
NormCSet = np.zeros(len(BesanconParamsDefined['BinName']),dtype=float)
if ModelParams['RecalculateNormConstants']:
    IntList = []
    for i in range(10):
        iBin = i + 1
        Int  = GetVolumeIntegral(iBin)
        IntList.append(Int)
        #print(Int)
    #Halo mass:
    IHalo            = np.where(BesanconParamsDefined['BinName'] == 'Halo')[0][0]
    NormCSet[IHalo]  = GalaxyParams['MHalo']/IntList[IHalo]
    #Bulge mass:
    IBulge           = np.where(BesanconParamsDefined['BinName'] == 'Bulge')[0][0]
    NormCSet[IBulge] = (GalaxyParams['MBulge'] + GalaxyParams['MBulge2'])/IntList[IBulge]
    #Thin/Thick disc masses:
    #First, get the non-weighted local densities
    RhoTildeSet      = np.array([RhoBesancon(GalaxyParams['RGalSun'], GalaxyParams['ZGalSun'], i + 1) for i in range(8)],dtype=float)        
    #Then, get the weights so that the local densities are reproduced
    NormSetPre       = BesanconParamsDefined['Rho0ParamSetMSunPcM3'][:8]/RhoTildeSet
    #Then renormalise the whole thin/thick disc to match the Galactic stellar mass and finalise the weights
    NormCSet[:8]     = NormSetPre*(GalaxyParams['MGal'] - (GalaxyParams['MBulge'] + GalaxyParams['MBulge2']) - GalaxyParams['MHalo'])/np.sum(NormSetPre*IntList[:8])
    
    #Compute derived quantities
    #Masses for each bin
    BinMasses        = NormCSet*IntList
    #Mass fractions in each bin
    BinMassFractions = BinMasses/GalaxyParams['MGal']
    
    NormConstantsDict = {'NormCSet': NormCSet, 'BinMasses': BinMasses, 'BinMassFractions': BinMassFractions}
    NormConstantsDF   = pd.DataFrame(NormConstantsDict)
    NormConstantsDF.to_csv('./Data/BesanconGalacticConstants.csv',index=False)
else:
    NormConstantsDF   = pd.read_csv('./Data/BesanconGalacticConstants.csv')
    NormConstantsDict = NormConstantsDF.to_dict(orient='list')
    
    
GalFunctionsDict = {'Besancon': RhoBesancon}

#Get the column density at a given radius for a given bin
def GetRhoBar(r,iBin,Model):
    Nz      = BesanconParamsDefined['ZNPoints'][iBin-1]
    ZRange  = BesanconParamsDefined['ZRange'][iBin-1]
    ZSet    = np.linspace(0,ZRange,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
    RhoBar  = np.sum(RhoSet)
    return RhoBar

# def GetZ(r,iBin,Model):
#     Nz      = 300
#     ZSet    = np.linspace(0,2,Nz)
#     RhoFun  = GalFunctionsDict[Model]
#     RhoSet  = np.zeros(Nz)
#     for i in range(Nz):
#         RhoSet[i] = RhoFun(r,ZSet[i],iBin)
#         
#     MidZSet    = 0.5*(ZSet[1:] + ZSet[:-1])
#     DeltaZSet  = 0.5*(ZSet[1:] - ZSet[:-1])
#     MidRhoSet  = 0.5*(RhoSet[1:] + RhoSet[:-1])
#     RhoBar     = np.sum(MidRhoSet*DeltaZSet)
#     RhozCDF    = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
#     
#     Xiz        = np.random.rand()
#     SignXi     = np.sign(2*(np.random.rand() - 0.5))
#     zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)    
#     return zFin


#A new version of GetZ - make a CDF for GetZ and save a grid of CDFs
def GetZCDF(r,iBin,Model):    
    Nz      = BesanconParamsDefined['ZNPoints'][iBin-1]
    ZRange  = BesanconParamsDefined['ZRange'][iBin-1]
    ZSet    = np.linspace(0,ZRange,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
        
    MidZSet    = 0.5*(ZSet[1:] + ZSet[:-1])
    DeltaZSet  = 0.5*(ZSet[1:] - ZSet[:-1])
    MidRhoSet  = 0.5*(RhoSet[1:] + RhoSet[:-1])
    RhoBar     = np.sum(MidRhoSet*DeltaZSet)
    RhozCDF    = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
    
    Res        = {'ZSet': MidZSet, 'RhoCDFSet': RhozCDF}
 
    return Res

#Part 2 of the draw z CDF function: using the earlier saved version of GetZ
def GetZ(RFin,iBin,Model):
    
    RSet = ModelRCache[iBin]['MidRSet']
    RID  = min(range(len(RSet)), key=lambda i: abs(RSet[i] - RFin))
    MidZSet = ZCDFDictSet[iBin+1][RID]['ZSet']
    RhozCDF = ZCDFDictSet[iBin+1][RID]['RhoCDFSet']
    
    Xiz        = np.random.rand()
    SignXi     = np.sign(2*(np.random.rand() - 0.5))
    zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)   
    return zFin
    
#Array of column densities as a function of radius
def RhoRArray(iBin, Model):
    Nr      = BesanconParamsDefined['RNPoints'][iBin-1]
    RRange  = BesanconParamsDefined['RRange'][iBin-1]
    RSet    = np.linspace(0,RRange,Nr)
    RhoSet  = np.zeros(Nr)
    ZCDFSet = {}
    for ir in range(Nr):
        RCurr       = RSet[ir]
        RhoSet[ir]  = GetRhoBar(RCurr,iBin,Model)
        #ZCDFSet[ir] = GetZCDF(RCurr,iBin,Model)
        
    Res = {'RSetKpc':RSet, 'RhoSet': RhoSet}
    return Res


def PreCompute(iBin, Model):
    RhoRDict   = RhoRArray(iBin, Model)
    RSet       = RhoRDict['RSetKpc']
    RhoRSet    = RhoRDict['RhoSet']
    MidRSet    = 0.5*(RSet[1:] + RSet[:-1])
    DeltaRSet  = RSet[1:] - RSet[:-1]
    MidRhoSet  = 0.5*(RhoRSet[1:] + RhoRSet[:-1])
    RhoNorm    = 2*np.pi*np.sum(MidRSet*DeltaRSet*MidRhoSet)
    RCDFSet    = 2*np.pi*np.cumsum(MidRSet*DeltaRSet*MidRhoSet)/RhoNorm
    
    Res        = {'MidRSet': MidRSet, 'RCDFSet': RCDFSet}
    return Res


#Import simulation files

#def print_structure(name, obj):
#    indent = '  ' * (name.count('/') - 1)
#    if isinstance(obj, h5py.Dataset):
#        print(f"{indent}- Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
#    elif isinstance(obj, h5py.Group):
#        print(f"{indent}- Group: {name}")

#Units
DaysToSec = float(str((u.d/u.s).decompose()))
YearToSec = float(str((u.yr/u.s).decompose()))
MyrToSec  = 1.e6*YearToSec
KmToCM    = float(str((u.km/u.cm).decompose()))
MSunToG   = ((const.M_sun/u.g).decompose()).value
RSunToCm  = ((const.R_sun/u.cm).decompose()).value

#Constants
GNewtCGS  = ((const.G*u.g*((u.s)**2)/(u.cm)**3).decompose()).value
CLightCGS = ((const.c*(u.s/u.cm)).decompose()).value
RGravSun  = 2.*GNewtCGS*MSunToG/CLightCGS**2
RhoConv   = (MSunToG/RSunToCm**3)

#GW constants
ADotGWPreFacCGS = (256./5)*((CLightCGS)**(-2))*(GNewtCGS*MSunToG/(RSunToCm*CLightCGS))**3
TauGWPreFacMyr  = (RSunToCm/ADotGWPreFacCGS)/MyrToSec

MRWDMset,MRWDRSet    = np.split(np.loadtxt(CodeDir + '/WDData/MRRel.dat'),2,axis=1)
MRWDMset             = MRWDMset.flatten()
MRWDRSet             = MRWDRSet.flatten()
MRSpl                = UnivariateSpline(MRWDMset, MRWDRSet, k=4, s=0)

#WD radius in RSun
def RWDPre(MWD):
    Res = float(MRSpl(MWD))
    return Res
RWD = np.vectorize(RWDPre)

#The orbital period in years
def POrbYrPre(MDonor, MAccretor, BinARSun):
    Omega = np.sqrt(GNewtCGS*MSunToG*(MDonor + MAccretor)/(BinARSun*RSunToCm)**3)
    Res   = (2.*np.pi/Omega)/YearToSec
    return Res
POrbYr = np.vectorize(POrbYrPre)

#The binary separation in RSun
def ABinRSunPre(MDonor, MAccretor, POrbYr):
    Omega = (2.*np.pi/(POrbYr*YearToSec))
    Res = ((GNewtCGS*MSunToG*(MDonor + MAccretor)/Omega**2)**(1/3))/RSunToCm
    return Res
ABinRSun = np.vectorize(ABinRSunPre)


#The GW inspiral time in megayears
def TGWMyrPre(M1MSun, M2MSun, aRSun):
    Res = TauGWPreFacMyr/((M1MSun + M2MSun)*M1MSun*M2MSun/aRSun**4)
    return Res
TGWMyr = np.vectorize(TGWMyrPre)

#The orbital separation after a given GW inspiral time
def APostGWRSunPre(M1MSun, M2MSun, AInitRSun, TGWInspMyr):
    TGWFull = TGWMyr(M1MSun, M2MSun, AInitRSun)
    Res     = AInitRSun*(1 - TGWInspMyr/TGWFull)**0.25
    return Res
APostGWRSun = np.vectorize(APostGWRSunPre)

#Roche lobe radius/BinA -- Eggeleton's formula
def fRL(q):
    X   = q**(1./3)
    Res = 0.49*(X**2)/(0.6*(X**2) + np.log(1.+X))
    return Res

#Roche lobe radius/BinA for the donor
def fRLDonor(MDonorMSun,MAccretorMSun):
    q = MDonorMSun/MAccretorMSun
    return fRL(q)



if ModelParams['ImportSimulation']:
    #Import data
    RunWave         = ModelParams['RunWave']
    RunSubType      = ModelParams['RunSubType']
    Code            = ModelParams['Code']
    if not Code == 'SEVN':
        FileName        = os.environ['UCB_GOOGLE_DRIVE_DIR'] + '/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_T0.hdf5'
    else:
        FileName        = os.environ['UCB_GOOGLE_DRIVE_DIR'] + '/simulated_binary_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/' + Code + '_MIST_T0.csv'
    CurrOutDir      = os.environ['UCB_GOOGLE_DRIVE_DIR'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + RunWave + '/' + RunSubType + '/'
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
    print('Step 1 done')

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
    
    print('Starting time convolution')    
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
            
            PresentDayDWDCandFin = pd.concat([PresentDayDWDCandFin.reset_index(drop=True), SubBinData.reset_index(drop=True)], axis=1)
            
            #Find present-day periods:
            PresentDayDWDCandFin['ATodayRSun']     = APostGWRSun(PresentDayDWDCandFin['mass1'], PresentDayDWDCandFin['mass2'], PresentDayDWDCandFin['semiMajor'], PresentDayDWDCandFin['SubBinMidAge'] - PresentDayDWDCandFin['time'])
            PresentDayDWDCandFin['PSetTodayHours'] = POrbYr(PresentDayDWDCandFin['mass1'],PresentDayDWDCandFin['mass2'], PresentDayDWDCandFin['ATodayRSun'])*YearToSec/(3600.)
            PresentDayDWDCandFin = PresentDayDWDCandFin.loc[PresentDayDWDCandFin['PSetTodayHours'] < 5.6]
            
            PresentDayDWDCandFinSet.append(PresentDayDWDCandFin)
            PresentDayDWDCandFin = []
                    
    PresentDayDWDCandFinDF = pd.concat(PresentDayDWDCandFinSet, ignore_index=True)
    

    print('Ending time convolution')
    
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
    



OutputChunkSize  = 100000
MultiProcessSize = 100
NChunks          = int(NGalDo/OutputChunkSize) + 1
NLeft            = NGalDo
FirstPassQ       = True

RSetFin  = np.zeros(OutputChunkSize)
ZSetFin  = np.zeros(OutputChunkSize)
ThSetFin = np.zeros(OutputChunkSize)
XSetFin  = np.zeros(OutputChunkSize)
YSetFin  = np.zeros(OutputChunkSize)
AgeFin   = np.zeros(OutputChunkSize)
BinFin   = np.zeros(OutputChunkSize)
FeHFin   = np.zeros(OutputChunkSize)

for Ch in range(NChunks):
    NDo  = min(NLeft,OutputChunkSize)
    
    if NDo < OutputChunkSize:
        RSetFin  = np.zeros(NDo)
        ZSetFin  = np.zeros(NDo)
        ThSetFin = np.zeros(NDo)
        XSetFin  = np.zeros(NDo)
        YSetFin  = np.zeros(NDo)
        AgeFin   = np.zeros(NDo)
        BinFin   = np.zeros(NDo)
        FeHFin   = np.zeros(NDo)
        
    
    #Draw all the needed positions for the chunk
    #First get the Bin ID for which we need the positions
    BinIDSet = PresentDayDWDCandFinDF.iloc[NGalDo-NLeft:NGalDo - NLeft + NDo]['BinID']
    
    #Then define a temporary function that draws positions for this chunk
    if ModelParams['ImportSimulation']:
        def DrawPos(i):
            return DrawStar('Besancon', int(BinIDSet.iloc[i]) + 1)
    else:
        ResCurr     = DrawStar('Besancon', -1)
        def DrawPos(i):
            return DrawStar('Besancon', int(BinIDSet.iloc[i]) + 1)
        
    iSet = [i for i in range(NDo)]
    
    #Tested parallel processing for this part - eventually it seems more effective to call the whole convolution per core
    #max_workers = os.cpu_count()
    
    #ChunkPosSet = []
    # 
    #with ProcessPoolExecutor(max_workers=max_workers) as exe:
    #    for res in exe.map(DrawPos, iSet, chunksize=MultiProcessSize):
    #        ChunkPosSet.append(res)
    
    DrawPosUse  = np.vectorize(DrawPos)
    ChunkPosSet = DrawPosUse(iSet)
    
    
    for i in range(NDo):
        if (NGalDo - NLeft + i) % OutputChunkSize == 0:
            print('Step 2: ', (NGalDo - NLeft + i), '/',NGalDo)
        if ModelParams['ImportSimulation']:
            #ResCurr     = DrawStar('Besancon', int(PresentDayDWDCandFinDF.iloc[NGalDo - NLeft + i]['BinID']) + 1)
            ResCurr     = ChunkPosSet[i]
            AgeFin[i]   = PresentDayDWDCandFinDF.iloc[NGalDo - NLeft + i]['SubBinMidAge']
        else:
            #ResCurr     = DrawStar('Besancon', -1)
            ResCurr     = ChunkPosSet[i]
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
    Gall     = np.zeros(NDo)
    Gall[YRel>=0] = np.arccos(XRel[YRel>=0]/(np.sqrt((RRel[YRel>=0])**2 - (ZRel[YRel>=0])**2)))
    Gall[YRel<0]  = 2*np.pi - np.arccos(XRel[YRel<0]/(np.sqrt((RRel[YRel<0])**2 - (ZRel[YRel<0])**2)))
    
    ResDict    = {'Bin': BinFin, 'Age': AgeFin, 'FeH': FeHFin, 'Xkpc': XSetFin, 'Ykpc': YSetFin, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel, 'Galb': Galb, 'Gall': Gall}
    ResDF      = pd.DataFrame(ResDict)
    MatchDWDDF = PresentDayDWDCandFinDF.iloc[NGalDo-NLeft:NGalDo-NLeft+NDo].reset_index(drop=True)
    
    #DWDDF    = DWDSet.iloc[IDSet]
    if ModelParams['ImportSimulation']:
        ResDF      = pd.concat([ResDF, MatchDWDDF], axis=1)
        if FirstPassQ:
            ResDF.to_csv(CurrOutDir+Code+'_Galaxy_AllDWDs.csv', index = False, mode ="w")
        else:
            ResDF.to_csv(CurrOutDir+Code+'_Galaxy_AllDWDs.csv', index = False, mode ="a", header=False)
    else:
        if FirstPassQ:
            ResDF.to_csv(CurrOutDir + '/FullGalaxyMSs.csv', index = False, mode ="w")
        else:
            ResDF.to_csv(CurrOutDir + '/FullGalaxyMSs.csv', index = False, mode ="a", header=False)

    ResDF      = []      
    NLeft     -= NDo 
    FirstPassQ = False
            
#Clean the memory-consuming arrays
PresentDayDWDCandFinDF = []
    
#Time the run
elapsed = time.perf_counter() - start
print('Convolution execution time: ', elapsed)
    

#Export only LISA-visible DWDs
if ModelParams['ImportSimulation']:
    
    ResDF = pd.read_csv(CurrOutDir+Code+'_Galaxy_AllDWDs.csv')#, usecols=['mass1', 'mass2', 'RRelkpc', 'PSetTodayHours'])
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



    
    