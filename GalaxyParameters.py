#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey
"""

import numpy as np


#Galaxy parameters
GalaxyParams = {'MGal': 6.43e10, #From Licquia and Newman 2015
                'MBulge': 6.1e9, #From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8, #From Robin+ 2012, metal-poor bulge
                'MHalo': 1.4e9, #From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'RGalSun': 8.122, #GRAVITY Collaboration et al. 2018
                'ZGalSun': 0.028 #Bennett & Bovy, 2019
               }


#3D components of the Sun’s velocity (U ; V ;W ) =(12:9; 245:6; 7:78) km s^1 (Drimmel & Poggio 2018)
