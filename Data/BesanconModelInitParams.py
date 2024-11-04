#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey
"""

import numpy as np


#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)

#Specify already known parameters
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


