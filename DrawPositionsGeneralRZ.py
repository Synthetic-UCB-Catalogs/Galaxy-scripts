#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import RegularGridInterpolator

from scipy.optimize import root
import sys
sys.path.insert(1, './PyModules/')

#Units are MSun, kpc, Gyr
#FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER


#Model parameters and options
ModelParams = {'GalaxyModel': 'Besancon',
               'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualizations
               'OneBinToUse': 10, #Number of the bin, if only one bin in used
               'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used (this option is not yet coded)
               'NPoints': 1e4 # Number of stars to sample
    }

#Galaxy model can be 'Besancon'


#Galaxy parameters
GalaxyParams = {'MGal': 6.43e10, #From Licquia and Newman 2015
                'MBulge': 6.1e9, #From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8, #From Robin+ 2012, metal-poor bulge
                'MHalo': 1.4e9, #From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'RGalSun': 8.2,  #Bland-Hawthorn, Gerhard 2016
                'ZGalSun': 0.025 #Bland-Hawthorn, Gerhard 2016
               }


######################################
#######    Model Specifications
###

#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)

#Define the model in two steps:
#First, specify already known parameters
BesanconParamsDefined = {
    'BinName':np.array(['ThinDisk1', 'ThinDisk2','ThinDisk3','ThinDisk4','ThinDisk5','ThinDisk6','ThinDisk7','ThickDisk','Halo','Bulge']),
    'AgeMin': 1000.*np.array([0,0.15,1,2,3,5,7,10,14,8],dtype='float64'),
    'AgeMax': 1000.*np.array([0.15,1.,2.,3.,5.,7.,10.,10.,14.,10],dtype='float64'),
    'XRange': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
    'YRange': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
    'ZRange': np.array([4,4,4,4,4,4,4,8,50,3],dtype='float64'),
    'FeHMean': np.array([0.01,0.03,0.03,0.01,-0.07,-0.14,-0.37,-0.78,-1.78,0.00],dtype='float64'),
    'FeHStD': np.array([0.12,0.12,0.10,0.11,0.18,0.17,0.20,0.30,0.50,0.40],dtype='float64'),
    #'Rho0ParamSetMSunPcM3': np.array([4.0e-3,7.9e-3,6.2e-3,4.0e-3,5.8e-3,4.9e-3,6.6e-3,1.34e-3,9.32e-6],dtype='float64'), Robin2003
    'Rho0ParamSetMSunPcM3': np.array([1.888e-3,5.04e-3,4.11e-3,2.84e-3,4.88e-3,5.02e-3,9.32e-3,2.91e-3,9.2e-6],dtype='float64'), #Czekaj2014
    'SigmaWKmS': np.array([6,8,10,13.2,15.8,17.4,17.5],dtype='float64'),
    'EpsSetThin': np.array([0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.0785, 0.0791],dtype='float64'),
    'EpsHalo': np.array([0.76],dtype='float64'),
    'dFedR': np.array([-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,0,0,0],dtype='float64'),
    'CSVNames':np.array(['GalTestThin1.csv','GalTestThin2.csv','GalTestThin3.csv','GalTestThin4.csv','GalTestThin5.csv','GalTestThin6.csv','GalTestThin7.csv','GalTestThick.csv','GalTestHalo.csv','GalTestBulge.csv'])
     }



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
    
def DrawStar(Model):
    BinSet = list(range(1,11))
    iBin   = np.random.choice(BinSet, p=BinMassFractions)
    RZ     = DrawRZ(iBin,Model)
    Age    = np.random.uniform(BesanconParamsDefined['AgeMin'][iBin-1],BesanconParamsDefined['AgeMax'][iBin-1])
    FeH    = np.random.normal(BesanconParamsDefined['FeHMean'][iBin-1],BesanconParamsDefined['FeHStD'][iBin-1])
    
    Res = {'RZ': RZ, 'Bin': iBin, 'Age': Age, 'FeH': FeH}

    return Res

RSetFin  = np.zeros(int(ModelParams['NPoints']))
ZSetFin  = np.zeros(int(ModelParams['NPoints']))
ThSetFin = np.zeros(int(ModelParams['NPoints']))
XSetFin  = np.zeros(int(ModelParams['NPoints']))
YSetFin  = np.zeros(int(ModelParams['NPoints']))
AgeFin   = np.zeros(int(ModelParams['NPoints']))
BinFin   = np.zeros(int(ModelParams['NPoints']))
FeHFin   = np.zeros(int(ModelParams['NPoints']))


for i in range(int(ModelParams['NPoints'])):
    ResCurr     = DrawStar('Besancon')
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
Gall     = np.zeros(int(ModelParams['NPoints']))
Gall[YRel>=0] = np.arccos(XRel[YRel>=0]/(np.sqrt((RRel[YRel>=0])**2 - (ZRel[YRel>=0])**2)))
Gall[YRel<0]  = 2*np.pi - np.arccos(XRel[YRel<0]/(np.sqrt((RRel[YRel<0])**2 - (ZRel[YRel<0])**2)))

ResDict  = {'Bin': BinFin, 'Age': AgeFin, 'FeH': FeHFin, 'Xkpc': XSetFin, 'Ykpc': YSetFin, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel, 'Galb': Galb, 'Gall': Gall}
ResDF    = pd.DataFrame(ResDict)    
ResDF.to_csv('./FullGalaxy.csv', index = False)




    
    
    
    