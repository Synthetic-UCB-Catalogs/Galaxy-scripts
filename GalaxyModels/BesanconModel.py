#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey
"""

import numpy as np
from utils import GetVolumeIntegral
from .GalaxyModelClass import GalaxyModel



#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)



# RTW what to do about these?
Alpha  = 78.9*(np.pi/180)
Beta   = 3.6*(np.pi/180)
Gamma  = 91.3*(np.pi/180)






#Define the model in two steps:
#First, specify already known parameters
#Specify already known parameters



class BesanconModel(GalaxyModel):
    # Redefine the necessary functions

    def GetModelParameters(self):
        # RTW: add units here
        BesanconParams = {
            'ComponentName':np.array(['ThinDisk1', 'ThinDisk2','ThinDisk3','ThinDisk4','ThinDisk5','ThinDisk6','ThinDisk7','ThickDisk','Halo','Bulge']),
            'AgeMin': 1000.*np.array([0,0.15,1,2,3,5,7,10,14,8],dtype='float64'),
            'AgeMax': 1000.*np.array([0.15,1.,2.,3.,5.,7.,10.,10.,14.,10],dtype='float64'),
            'XMax': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
            'YMax': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
            'ZMax': np.array([4,4,4,4,4,4,4,8,50,3],dtype='float64'),
            'FeHMean': np.array([0.01,0.03,0.03,0.01,-0.07,-0.14,-0.37,-0.78,-1.78,0.00],dtype='float64'),
            'FeHStD': np.array([0.12,0.12,0.10,0.11,0.18,0.17,0.20,0.30,0.50,0.40],dtype='float64'),
            #'Rho0ParamSetMSunPcM3': np.array([4.0e-3,7.9e-3,6.2e-3,4.0e-3,5.8e-3,4.9e-3,6.6e-3,1.34e-3,9.32e-6],dtype='float64'), Robin2003
            'Rho0ParamSetMSunPcM3': np.array([1.888e-3,5.04e-3,4.11e-3,2.84e-3,4.88e-3,5.02e-3,9.32e-3,2.91e-3,9.2e-6],dtype='float64'), #Czekaj2014
            'SigmaWKmS': np.array([6,8,10,13.2,15.8,17.4,17.5],dtype='float64'),
            'EpsSetThin': np.array([0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.0785, 0.0791],dtype='float64'),
            'EpsHalo': np.array([0.76],dtype='float64'),
            'dFedR': np.array([-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,0,0,0],dtype='float64'),
            'CSVNames':np.array(['GalTestThin1.csv','GalTestThin2.csv','GalTestThin3.csv','GalTestThin4.csv','GalTestThin5.csv','GalTestThin6.csv','GalTestThin7.csv','GalTestThick.csv','GalTestHalo.csv','GalTestBulge.csv']),
            #'RhoFunction': RhoBesancon,
            'NormedMassFile': 'BesanconGalacticConstants.csv',
             }
        return BesanconParams


    def RhoFunction(self, r, z, iComponent):
        #Rho(r,z), for the Besancon model, weights are defined later

        global Alpha, Beta, Gamma
        #print(iComponent)
        #Young thin disc
        if iComponent == 0:
            hPlus  = 5
            hMinus = 3
            Eps    = BesanconParams['EpsSetThin'][0]
            aParam = np.sqrt(r**2 + z**2/Eps**2)
            Rho    = np.exp(-(aParam**2/hPlus**2)) - np.exp(-(aParam**2/hMinus**2))
        #Thin disk - other bins
        elif (iComponent >= 1) and (iComponent <=6):
            hPlus  = 2.53
            hMinus = 1.32
            Eps    = BesanconParams['EpsSetThin'][iComponent]
            aParam = np.sqrt(r**2 + z**2/Eps**2)
            Rho    = np.exp(-(0.5**2 + aParam**2/hPlus**2)**0.5) - np.exp(-(0.5**2 + aParam**2/hMinus**2)**0.5)
        #Thick disc
        elif (iComponent == 7):
            xl   = 0.4
            RSun = 8
            hR   = 2.5
            hz   = 0.8
            Rho  = np.where(np.abs(z) <=xl, (np.exp(-(r-RSun)/hR))*(1.-((1/hz)/(xl*(2+xl/hz)))*(z**2)),
                            (np.exp(-(r-RSun)/hR))*((np.exp(xl/hz))/(1+xl/(2*hz)))*(np.exp(-np.abs(z)/hz)))
        #Halo
        elif (iComponent == 8):
            ac   = 0.5
            Eps  = 0.76
            RSun = 8. 
            aParam = np.sqrt(r**2 + z**2/Eps**2)
            Rho    = np.where((aParam<=ac), (ac/RSun)**(-2.44),(aParam/RSun)**(-2.44))
        #Bulge
        elif (iComponent == 9):
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
            Rho    = np.where(rParam<=Rc, np.exp(-0.5*rs2),
                #np.exp(-0.5*rs2)*np.exp(-0.5*((np.sqrt(x**2 + y**2) - Rc)/(0.5))**2)
                np.exp(-0.5*rs2)*np.exp(-0.5*((rbulge - Rc)/(0.5))**2))
            
        return Rho

    def CalculateGalacticComponentMassFractions(self):

        IntList = []
        for iComponent in range(10):
            Int  = GetVolumeIntegral( GalaxyModelParams['XMax'][iComponent],
                                      GalaxyModelParams['YMax'][iComponent],
                                      GalaxyModelParams['ZMax'][iComponent],
                                      self.RhoFunction)

            IntList.append(Int)
            #print(Int)
        #Halo mass:
        IHalo              = np.where(GalaxyModelParams['ComponentName'] == 'Halo')[0][0]
        NormCArray[IHalo]  = MWConsts['MHalo']/IntList[IHalo]
        #Bulge mass:
        IBulge             = np.where(GalaxyModelParams['ComponentName'] == 'Bulge')[0][0]
        NormCArray[IBulge] = MWConsts['MBulge']/IntList[IBulge]
        #Thin/Thick disc masses:
        #First, get the non-weighted local densities
        RhoTildeArray      = np.array([self.RhoFunction(MWConsts['RGalSun'], MWConsts['ZGalSun'], i) for i in range(8)],dtype=float)        
        #Then, get the weights so that the local densities are reproduced
        NormArrayPre       = GalaxyModelParams['Rho0ParamArrayMSunPcM3'][:8]/RhoTildeArray
        #Then renormalise the whole thin/thick disc to match the Galactic stellar mass and finalise the weights
        NormCArray[:8]     = NormArrayPre*(MWConsts['MGal'] - MWConsts['MBulge'] - MWConsts['MHalo'])/np.sum(NormArrayPre*IntList[:8])
        
        #Compute derived quantities
        #Masses for each bin
        ComponentMasses    = NormCArray*IntList
        #Mass fractions in each bin
        ComponentMassFractions = ComponentMasses/MWConsts['MGal']
        
        NormConstantsDict = {'NormCArray': NormCArray, 'ComponentMasses': ComponentMasses, 'ComponentMassFractions': ComponentMassFractions}
        NormConstantsDF   = pd.DataFrame(NormConstantsDict)
        NormConstantsDF.to_csv('GalacticModelConstants.csv',index=False)

        return 

    ####################




    def GetRhoBar(self, r,iComponent,Model):
        Nz      = 300
        ZSet    = np.linspace(0,2,Nz)
        RhoArray  = np.zeros(Nz)
        for i in range(Nz):
            RhoArray[i] = self.RhoFunction(r, ZArray[i], iComponent)
        RhoBar  = np.sum(RhoArray)
        return RhoBar

    #A new version of GetZ - make a CDF for GetZ and save a grid of CDFs
    def GetZCDF(self,r,iComponent,Model):    
        Nz      = 300
        ZSet    = np.linspace(0,2,Nz)
        RhoFun  = GalFunctionsDict[Model]
        RhoSet  = np.zeros(Nz)
        for i in range(Nz):
            RhoSet[i] = RhoFun(r,ZSet[i],iComponent)
            
        MidZSet    = 0.5*(ZSet[1:] + ZSet[:-1])
        DeltaZSet  = 0.5*(ZSet[1:] - ZSet[:-1])
        MidRhoSet  = 0.5*(RhoSet[1:] + RhoSet[:-1])
        RhoBar     = np.sum(MidRhoSet*DeltaZSet)
        RhozCDF    = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
        
        Res        = {'ZSet': MidZSet, 'RhoCDFSet': RhozCDF}
     
        return Res
    
    #Part 2 of the draw z CDF function: using the earlier saved version of GetZ
    def GetZ(self, RFin,iComponent,Model):
        
        RSet = ModelRCache[iComponent]['MidRSet']
        RID  = min(range(len(RSet)), key=lambda i: abs(RSet[i] - RFin))
        MidZSet = ZCDFDictSet[iComponent+1][RID]['ZSet']
        RhozCDF = ZCDFDictSet[iComponent+1][RID]['RhoCDFSet']
        
        Xiz        = np.random.rand()
        SignXi     = np.sign(2*(np.random.rand() - 0.5))
        zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)   
        return zFin
        
    
    def RhoRArray(self, iComponent, Model):
        Nr      = 1000
        RSet    = np.linspace(0,30,Nr)
        RhoSet  = np.zeros(Nr)
        ZCDFSet = {}
        for ir in range(Nr):
            RCurr       = RSet[ir]
            RhoSet[ir]  = GetRhoBar(RCurr,iComponent,Model)
            #ZCDFSet[ir] = GetZCDF(RCurr,iComponent,Model)
            
        Res = {'RSetKpc':RSet, 'RhoSet': RhoSet}
        return Res
    
    
    def PreCompute(self, iComponent, Model):
        RhoRDict   = RhoRArray(iComponent, Model)
        RSet       = RhoRDict['RSetKpc']
        RhoRSet    = RhoRDict['RhoSet']
        MidRSet    = 0.5*(RSet[1:] + RSet[:-1])
        DeltaRSet  = RSet[1:] - RSet[:-1]
        MidRhoSet  = 0.5*(RhoRSet[1:] + RhoRSet[:-1])
        RhoNorm    = 2*np.pi*np.sum(MidRSet*DeltaRSet*MidRhoSet)
        RCDFSet    = 2*np.pi*np.cumsum(MidRSet*DeltaRSet*MidRhoSet)/RhoNorm
        
        Res        = {'MidRSet': MidRSet, 'RCDFSet': RCDFSet}
        return Res



    def CalculateRZ_CDFs(self):
        return


        #Get the R-CDFs
        #if ModelParams['RecalculateCDFs']: 
        
        #ModelCache     = PreCompute(ModelParams['OneComponentToUse'],'Besancon')
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
        iComponentSampleSet = [i for i in range(10)]
    
        # Create another HDF5 file
        with h5py.File('./GalCache/BesanconRZData.h5', 'w') as hdf5_file:
            for iComponent in iComponentSampleSet:
                print('Caching Component ' + str(iComponent+1))
                # Create a group for each x value
                x_group = hdf5_file.create_group(f'bin_{iComponent+1}')
                rSet    = ModelRCache[iComponent]['MidRSet']
                rIDs    = list(range(len(rSet)))
                for rID in rIDs:
                    if (rID % 100) == 0:
                        print('rID '+ str(rID))
                    # Create a subgroup for each y value within the x group
                    y_group = x_group.create_group(f'r_{rID}')
                    # Compute the function output
                    data_dict = GetZCDF(rSet[rID], iComponent + 1,'Besancon')
                    # Store each list in the dictionary as a dataset
                    for key, value in data_dict.items():
                        y_group.create_dataset(key, data=value, compression='gzip')




    ######################################################
    ############ Galaxy Sampling Part
    ####    
    
    
    #Routine to load data from an 1-D organised hdf5 file
    
    
    
    
    #Get the z-CDFs
    
    def DrawRZ(iComponent,Model):
        MidRSet    = ModelRCache[iComponent-1]['MidRSet']
        RCDFSet    = ModelRCache[iComponent-1]['RCDFSet']
        
        Xir        = np.random.rand()
        RFin       = np.interp(Xir,RCDFSet,MidRSet)
        zFin       = GetZ(RFin,iComponent-1,Model)
        
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
        
    def DrawStar(Model, iComponent):
        if iComponent == -1:
            ComponentSet = list(range(10))
            iComponent   = np.random.choice(ComponentSet, p=galacticComponentMassFractions)
        RZ     = DrawRZ(iComponent,Model)
        Age    = np.random.uniform(GalaxyModelParams['AgeMin'][iComponent],GalaxyModelParams['AgeMax'][iComponent])
        FeH    = np.random.normal(GalaxyModelParams['FeHMean'][iComponent],GalaxyModelParams['FeHStD'][iComponent])
        
        Res = {'RZ': RZ, 'Component': iComponent, 'Age': Age, 'FeH': FeH}
    
        return Res














    #else:
    #    #Load the previously calculated r CDFs
    #    ModelRCache     = []
    #    for Dict in load_Rdicts_from_hdf5('./GalCache/BesanconRData.h5'):
    #        # Process each dictionary one at a time
    #        ModelRCache.append(Dict)        
    #    #Load the previously calculated rz CDFs
    #    ZCDFDictSet = load_RZdicts_from_hdf5('./GalCache/BesanconRZData.h5')





    #def GetZ(r,iComponent,Model):
    #    Nz      = 300
    #    ZSet    = np.linspace(0,2,Nz)
    #    RhoFun  = GalFunctionsDict[Model]
    #    RhoSet  = np.zeros(Nz)
    #    for i in range(Nz):
    #        RhoSet[i] = RhoFun(r,ZSet[i],iComponent)
    #        
    #    MidZSet    = 0.5*(ZSet[1:] + ZSet[:-1])
    #    DeltaZSet  = 0.5*(ZSet[1:] - ZSet[:-1])
    #    MidRhoSet  = 0.5*(RhoSet[1:] + RhoSet[:-1])
    #    RhoBar     = np.sum(MidRhoSet*DeltaZSet)
    #    RhozCDF    = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
    #    
    #    Xiz        = np.random.rand()
    #    SignXi     = np.sign(2*(np.random.rand() - 0.5))
    #    zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)    
    #    return zFin
