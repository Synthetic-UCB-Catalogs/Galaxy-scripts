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

#Units are kpc, Gyr


def zCDFInv(Xiz,Hz):
    zCoord = -Hz*np.log(1-Xiz)
    return zCoord



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

 
def Sample1DPop(NComponent,Hr,Hz):
    RRandSet = np.random.uniform(0, 1, NComponent)
    ZRandSet = np.random.uniform(0, 1, NComponent)
    ZSignSet = np.sign(np.random.uniform(0, 2, NComponent) - 1)
    RSet     = np.asarray([RCDFInv(RRandSet[i],Hr) for i in range(NComponent)],dtype=float)
    ZSet     = np.asarray([zCDFInv(ZRandSet[i],Hz) for i in range(NComponent)],dtype=float)*ZSignSet
    ThSet    = np.random.uniform(0, 2.*np.pi, NComponent)
    XSet     = RSet*np.cos(ThSet)
    YSet     = RSet*np.sin(ThSet)
    AgeSet   = np.random.uniform(0, 12, NComponent)
    IDSet    = np.arange(NComponent) + 1    
    
    ResDict  = {'ID':IDSet, 'Ages': AgeSet, 'Xkpc': XSet, 'Ykpc': YSet, 'Zkpc': ZSet, 'Rkpc': RSet, 'Th': ThSet}
    ResDF    = pd.DataFrame(ResDict)    
    return ResDF

ExportTable = False

if ExportTable:
    NComponent = 10**4
    Hr   = 4
    Hz   = 0.5
    
    Res = Sample1DPop(NComponent,Hr,Hz)
    
    Res.to_csv('./GalTest.csv', index = False)
                
    
    
    
    
