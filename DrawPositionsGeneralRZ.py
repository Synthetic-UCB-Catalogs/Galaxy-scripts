#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import root
import sys
sys.path.insert(1, './PyModules/')

#Units are MSun, kpc, Gyr

MGal = 6.43e10 #From Licquia and Newman 2015
iBinUse = 8

#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php
#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)
BesanconParamsPre = {#'Rho0Set0': [0.00188],
                     #'Rho0SetThin': [0.00504, 0.00411, 0.00284, 0.00488, 0.00502, 0.00932], 
                     #'Rho0Thick': [0.00291], 
                     #'Rho0Halo': [0.000092],
                     'EpsBin1': [0.0140],
                     'EpsSetThin': [0.0268,0.0375, 0.0551, 0.0696, 0.0785, 0.0791],
                     'EpsThick': [],
                     'EpsHalo': [0.76], 
                     'MFracPreNonNorm': [0.00188, 0.00504, 0.00411, 0.00284, 0.00488, 0.00502, 0.00932, 0.00291, 0.000092]
                     }



GalDict = {
        'Name':np.array(['ThinDisk1', 'ThinDisk2','ThinDisk3','ThinDisk4','ThinDisk5','ThinDisk6','ThinDisk7','ThickDisk','Halo','Bulge']),
        'AgeMin': 1000.*np.array([0,0.15,1,2,3,5,7,10,14,8],dtype='float64'),
        'AgeMax': 1000.*np.array([0.15,1.,2.,3.,5.,7.,10.,10.,14.,10],dtype='float64'),
        'FeHMean': np.array([0.01,0.03,0.03,0.01,-0.07,-0.14,-0.37,-0.78,-1.78,0.00],dtype='float64'),
        'FeHStD': np.array([0.12,0.12,0.10,0.11,0.18,0.17,0.20,0.30,0.50,0.40],dtype='float64'),
        #'MFrac':np.array([0.0210924, 0.0481609, 0.0528878, 0.0501353, 0.0918268, 0.087498, 0.118755, 0.0855393, 7.49753*10.**(-8), 0.444104],dtype='float64')
        'MFrac':np.array([0.0303435, 0.0692841, 0.0760842, 0.0721246, 0.132102, 0.125874, 0.170841, 0.123057, 0.00830868, 0.191981]), #Global Galaxy
        #'MFrac':np.array([0.0981611, 0.193868, 0.15215, 0.0981611, 0.142334, 0.120247,0.161966, 0.032884, 0.000228715, 0.]), #Galaxy local
        #'MFrac':np.array([0.0968262, 0.191232, 0.150081, 0.0968262, 0.140398, 0.118612,0.159763, 0.0458724, 0.000389008, 0.]), #Global Galaxy 500pc
        #'MFrac':np.array([0.0956799, 0.188968, 0.148304, 0.0956799, 0.138736, 0.117208,0.157872, 0.0569753, 0.000577695, 0.]), #Global Galaxy 1000pc
        'CSVNames':['GalTestThin1.csv','GalTestThin2.csv','GalTestThin3.csv','GalTestThin4.csv','GalTestThin5.csv','GalTestThin6.csv','GalTestThin7.csv','GalTestThick.csv','GalTestHalo.csv','GalTestBulge.csv'],
        'MTot':3.15*(10**10) #From original Besancon, 6.43 10^10 - more modern
        }

#Rho(r,z), non-normalized for the Besancon model, thin disk
def RhoBesanconPre(r,z,iBin):
    #print(iBin)
    #Young thin disc
    if iBin == 1:
        hPlus  = 5
        hMinus = 3
        Eps    = BesanconParamsPre['EpsBin1'][0]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(aParam**2/hPlus**2)) - np.exp(-(aParam**2/hMinus**2))
    #Thin disk - other bins
    elif (iBin >= 2) and (iBin <=7):
        hPlus  = 2.53
        hMinus = 1.32
        Eps    = BesanconParamsPre['EpsSetThin'][iBin - 2]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(0.5**2 + aParam**2/hPlus**2)**0.5) - np.exp(-(0.5**2 + aParam**2/hMinus**2)**0.5)
    #Thick disc
    elif (iBin == 8):
        xl   = 0.4
        RSun = 8
        hR   = 2.5
        hz   = 0.8
        if np.abs(z) <=xl:
            Res = (np.exp(-(r-RSun)/hR))*(1.-((1/hz)/(xl*(2+xl/hz)))*(z**2))
        else:
            Res = (np.exp(-(r-RSun)/hR))*((np.exp(xl/hz))/(1+xl/(2*hz)))*(np.exp(-np.abs(z)/hz))
    #Halo
    elif (iBin == 9):
        ac   = 0.5
        Eps  = 0.76
        RSun = 8. 
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        if (aParam<=ac):
            Res = (ac/RSun)**(-2.44)
        else:
            Res = (aParam/RSun)**(-2.44)
    #Bulge
    elif (iBin == 10):
        x0 = 1.59
        y0 = 0.424
        z0 = 0.424
        Rc = 2.54
        N  = 13.7
        #See Robin Tab 5
        #Orientation angles: α (angle between the bulge major axis and the line perpendicular to the Sun – Galactic Center line), 
        #β (tilt angle between the bulge plane and the Galactic plane) and 
        #γ (roll angle around the bulge major axis);
        Alpha  = 78.9*(np.pi/180)
        Beta   = 3.6*(np.pi/180)
        Gamma  = 91.3*(np.pi/180)
        #Implement in 3D CHECK ONCE 3D IS BEING IMPLEMENTED
        #NOTE, EXPRESSION is NOT FINAL
        x      = r*np.cos(Alpha)
        y      = r*np.sin(Alpha)
        rs2    = np.sqrt(((x/x0)**2 + (y/y0)**2)**2 + (z/z0)**4)                
        rParam = np.sqrt(x**2 + y**2)
        if rParam<=Rc:
            Res = np.exp(-0.5*rs2)
        else:
            Res = np.exp(-0.5*rs2)*np.exp(-0.5*((np.sqrt(x**2 + y**2) - Rc)/(0.5))**2)
        

    return Res

GalFunctionsDict = {'Besancon': RhoBesanconPre}

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

ModelCache     = PreCompute(iBinUse,'Besancon')
    
def DrawRZ(iBin,Model):
    MidRSet    = ModelCache['MidRSet']
    RCDFSet    = ModelCache['RCDFSet']
    
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
    
NPoints = 10000

RSetFin = np.zeros(NPoints)
ZSetFin = np.zeros(NPoints)

for i in range(NPoints):
    ResCurr     = DrawRZ(iBinUse,'Besancon')
    RSetFin[i]  = ResCurr[0]
    ZSetFin[i]  = ResCurr[1]
    

Th       = np.asarray([2.*np.pi*np.random.uniform() for i in range(NPoints)],dtype=float)
X        = RSetFin*np.cos(Th)
Y        = RSetFin*np.sin(Th)
Age      = np.random.uniform(0, 12)
 
XRel     = X - 8.
YRel     = Y
ZRel     = ZSetFin

RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)

ResDict  = {'Age': Age, 'Xkpc': X, 'Ykpc': Y, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel}
ResDF    = pd.DataFrame(ResDict)    
ResDF.to_csv('./' + GalDict['CSVNames'][iBinUse-1], index = False)

sys.exit()




def Sample1D(Hr,Hz):
    RRand    = np.random.uniform()
    ZRand    = np.random.uniform()
    ZSign    = np.sign(np.random.uniform() - 0.5)
    R        = RCDFInv(RRand,Hr)
    Z        = zCDFInv(ZRand,Hz)*ZSign
    Th       = 2.*np.pi*np.random.uniform()
    X        = R*np.cos(Th)
    Y        = R*np.sin(Th)
    Age      = np.random.uniform(0, 12)
    
    XRel     = X - 8.
    YRel     = Y
    ZRel     = Z

    RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)

    ResDict  = {'Age': Age, 'Xkpc': X, 'Ykpc': Y, 'Zkpc': Z, 'Rkpc': R, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel}
    
    return ResDict

 
def Sample1DPop(NBin,Hr,Hz):
    RRandSet = np.random.uniform(0, 1, NBin)
    ZRandSet = np.random.uniform(0, 1, NBin)
    ZSignSet = np.sign(np.random.uniform(0, 2, NBin) - 1)
    RSet     = np.asarray([RCDFInv(RRandSet[i],Hr) for i in range(NBin)],dtype=float)
    ZSet     = np.asarray([zCDFInv(ZRandSet[i],Hz) for i in range(NBin)],dtype=float)*ZSignSet
    ThSet    = np.random.uniform(0, 2.*np.pi, NBin)
    XSet     = RSet*np.cos(ThSet)
    YSet     = RSet*np.sin(ThSet)
    AgeSet   = np.random.uniform(0, 12, NBin)
    IDSet    = np.arange(NBin) + 1    
    
    ResDict  = {'ID':IDSet, 'Ages': AgeSet, 'Xkpc': XSet, 'Ykpc': YSet, 'Zkpc': ZSet, 'Rkpc': RSet, 'Th': ThSet}
    ResDF    = pd.DataFrame(ResDict)    
    return ResDF

ExportTable = False

if ExportTable:
    NBin = 10**4
    Hr   = 4
    Hz   = 0.5
    
    Res = Sample1DPop(NBin,Hr,Hz)
    
    Res.to_csv('./' + GalDict['CSVNames'][iBinUse-1], index = False)
                
    
    
    
    