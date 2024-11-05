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
from BesanconModelInitParams import BesanconParamsDefined
from GalaxyParameters import GalaxyParams
from get_mass_norm import get_mass_norm
from rapid_code_load_T0 import load_T0_data

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER


#Model parameters and options
ModelParams = {'GalaxyModel': 'Besancon',
               'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualizations
               'OneBinToUse': 10, #Number of the bin, if only one bin in used
               'RecalculateNormConstants': False, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)
               'NPoints': 1e4 # Number of stars to sample if we just sample present-day stars
    }

#Galaxy model can be 'Besancon'



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
        Alpha  = 78.9*(np.pi/180)
        Beta   = 3.6*(np.pi/180)
        Gamma  = 91.3*(np.pi/180)

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
#     NPoints = 400
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
    NPoints = 1000
    
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
    


def GetRhoBar(r,iBin,Model):
    Nz      = 300
    ZSet    = np.linspace(0,2,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
    RhoBar  = np.sum(RhoSet)
    return RhoBar

def GetZ(r,iBin,Model):
    Nz      = 300
    ZSet    = np.linspace(0,2,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
        
    MidZSet   = 0.5*(ZSet[1:] + ZSet[:-1])
    DeltaZSet = 0.5*(ZSet[1:] - ZSet[:-1])
    MidRhoSet = 0.5*(RhoSet[1:] + RhoSet[:-1])
    RhoBar    = np.sum(MidRhoSet*DeltaZSet)
    RhozCDF   = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
    
    Xiz        = np.random.rand()
    SignXi     = np.sign(2*(np.random.rand() - 0.5))
    zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)    
    return zFin
    
           

def RhoRArray(iBin, Model):
    Nr     = 1000
    RSet   = np.linspace(0,30,Nr)
    RhoSet = np.zeros(Nr)
    for i in range(Nr):
        RCurr      = RSet[i]
        RhoSet[i]  = GetRhoBar(RCurr,iBin,Model)
        
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
    #Parameters
    RunWave         = 'fiducial'
    FileName        = './Simulations/COSMIC_T0.hdf5'
    ACutRSunPre     = 6     #Initial cut for all binaries
    LISAPCutHours   = 0.5*(1/1.e-4)/(3600.)  #1/e-4 Hz + remember that GW frequency is 2X the orbital frequency
    MaxTDelay       = 14000    
    RepresentDWDsBy = 100000   #Represent the present-day LISA candidates by this nubmer of binaries
    DeltaTGalMyr    = 50       #Time step resolution in the Galactic SFR
    
    #General quantities
    MassNorm        = get_mass_norm(RunWave)
    NStarsPerRun    = GalaxyParams['MGal']/MassNorm
    SimData         = load_T0_data(FileName)
    NRuns           = SimData[1]['NSYS'][0]
    
    #Pre-process simulations
    Sims                          = SimData[0]
    DWDSetPre                     = Sims.loc[(Sims.type1.isin([21,22,23])) & (Sims.type2.isin([21,22,23])) & (Sims.semiMajor > 0) & (Sims.semiMajor < ACutRSunPre)].groupby('ID', as_index=False).first() #DWD binaries at the moment of formation with a<8RSun
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
    SubBinDF.to_csv('./GalaxyLISACandidateStats.csv', index = False)
    
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
    NFindPre     = RepresentDWDsBy
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
            
    
######################################################
############ Galaxy Sampling Part
####    

#ModelCache     = PreCompute(ModelParams['OneBinToUse'],'Besancon')
ModelCache     = {}
for i in range(10):
    ModelCache[i] = PreCompute(i+1,'Besancon')
    

#GalDict = {##'MFrac':np.array([0.0210924, 0.0481609, 0.0528878, 0.0501353, 0.0918268, 0.087498, 0.118755, 0.0855393, 7.49753*10.**(-8), 0.444104],dtype='float64')
           ##'MFracPreNonNorm': [0.00188, 0.00504, 0.00411, 0.00284, 0.00488, 0.00502, 0.00932, 0.00291, 0.000092]
        #'MFrac':np.array([0.0303435, 0.0692841, 0.0760842, 0.0721246, 0.132102, 0.125874, 0.170841, 0.123057, 0.00830868, 0.191981]), #Global Galaxy
        ##'MFrac':np.array([0.0981611, 0.193868, 0.15215, 0.0981611, 0.142334, 0.120247,0.161966, 0.032884, 0.000228715, 0.]), #Galaxy local
        ##'MFrac':np.array([0.0968262, 0.191232, 0.150081, 0.0968262, 0.140398, 0.118612,0.159763, 0.0458724, 0.000389008, 0.]), #Global Galaxy 500pc
        ##'MFrac':np.array([0.0956799, 0.188968, 0.148304, 0.0956799, 0.138736, 0.117208,0.157872, 0.0569753, 0.000577695, 0.]), #Global Galaxy 1000pc
        #}


def DrawRZ(iBin,Model):
    MidRSet    = ModelCache[iBin-1]['MidRSet']
    RCDFSet    = ModelCache[iBin-1]['RCDFSet']
    
    Xir        = np.random.rand()
    RFin       = np.interp(Xir,RCDFSet,MidRSet)
    zFin       = GetZ(RFin,iBin,Model)
    
    return [RFin,zFin]

    def RCDFInv(Xir,Hr):    
        # Get the parameters for the inverse CDF
        def RCD(R):
            Res = (1-np.exp(-R/Hr))-(R/Hr)*np.exp(-R/Hr)-Xir
            return Res
        
        Sol  = sp.optimize.root_scalar(RCD,bracket=(0.0001*Hr,20*Hr))
        if Sol.converged:
            R      = Sol.root
        else:
            print('The radial solution did not converge')
            sys.exit()
        return R
    
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

ResDF.to_csv('./FullGalaxyDWD.csv', index = False)

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
    
    LISADF.to_csv('./FullGalaxyDWDLISAOnly.csv', index = False)


    
    
    
    