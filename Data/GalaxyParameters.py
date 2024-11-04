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
                'RGalSun': 8.2, #Bland-Hawthorn, Gerhard 2016
                'ZGalSun': 0.025 #Bland-Hawthorn, Gerhard 2016
               }