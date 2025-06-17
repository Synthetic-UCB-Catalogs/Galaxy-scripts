######################################
#######    Galactic Model Specifications
###

#For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

#Here we use:
#1) Eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
#2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)

#Define the model in two steps:
#First, specify already known parameters

import numpy as np




#Non-normalized density (Rho(r,z)) for the Besancon model
# weights are defined later
def RhoBesancon(r, z, iBin, BesanconParamsDefined):
    #Young thin disc
    if iBin == 0:
        hPlus  = 5
        hMinus = 3
        Eps    = BesanconParamsDefined['EpsSetThin'][0]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(aParam**2/hPlus**2)) - np.exp(-(aParam**2/hMinus**2))
    #Thin disk - other bins
    elif (iBin >= 1) and (iBin <=6):
        hPlus  = 2.53
        hMinus = 1.32
        Eps    = BesanconParamsDefined['EpsSetThin'][iBin - 1]
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.exp(-(0.5**2 + aParam**2/hPlus**2)**0.5) - np.exp(-(0.5**2 + aParam**2/hMinus**2)**0.5)
    #Thick disc
    elif (iBin == 7):
        xl   = 0.4
        RSun = 8
        hR   = 2.5
        hz   = 0.8
        Res  = np.where(np.abs(z) <=xl, (np.exp(-(r-RSun)/hR))*(1.-((1/hz)/(xl*(2+xl/hz)))*(z**2)),
                        (np.exp(-(r-RSun)/hR))*((np.exp(xl/hz))/(1+xl/(2*hz)))*(np.exp(-np.abs(z)/hz)))
    #Halo
    elif (iBin == 8):
        ac   = 0.5
        Eps  = 0.76
        RSun = 8. 
        aParam = np.sqrt(r**2 + z**2/Eps**2)
        Res    = np.where((aParam<=ac), (ac/RSun)**(-2.44),(aParam/RSun)**(-2.44))
    #Bulge
    elif (iBin == 9):
        x0 = 1.59
        yz0 = 0.424
        #yz0 = 0.424 -- y0 and z0 are equal, use that to sample the bulge 
        #stars in the coordinates where z-axis is aligned with the x-axis of the bugle
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

#2D Volume integrator for the Galactic density components
def GetVolumeIntegral(iBin, BesanconParamsDefined):
    NPoints =  BesanconParamsDefined['ZNPoints'][iBin]
    
    RRange = np.sqrt((BesanconParamsDefined['XRange'][iBin])**2 + (BesanconParamsDefined['YRange'][iBin])**2)
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

#Get the column density at a given radius for a given bin
def GetRhoBar(r, iBin, Model, BesanconParamsDefined):
    Nz      = BesanconParamsDefined['ZNPoints'][iBin-1]
    ZRange  = BesanconParamsDefined['ZRange'][iBin-1]
    ZSet    = np.linspace(0,ZRange,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
    RhoBar  = np.sum(RhoSet)
    return RhoBar


#A new version of GetZ - make a CDF for GetZ and save a grid of CDFs
def GetZCDF(r,iBin,Model, BesanconParamsDefined):    
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
def RhoRArray(iBin, Model, BesanconParamsDefined):
    Nr      = BesanconParamsDefined['RNPoints'][iBin-1]
    RRange  = BesanconParamsDefined['RRange'][iBin-1]
    RSet    = np.linspace(0,RRange,Nr)
    RhoSet  = np.zeros(Nr)
    ZCDFSet = {}
    for ir in range(Nr):
        RCurr       = RSet[ir]
        RhoSet[ir]  = GetRhoBar(RCurr,iBin,Model)
        
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


if __name__=="__main__":
    
    import numpy as np
    from BesanconModelInitParams import BesanconParamsDefined
    from GalaxyParameters import GalaxyParams

    
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
               'UseRepresentingWDs': False, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, RepresentDWDsBy, binaries
               'RepresentDWDsBy': 500000,  #Represent the present-day LISA candidates by this nubmer of binaries
               'LISAPCutHours': (2/1.e-4)/(3600.), #LISA cut-off orbital period, 1.e-4 Hz + remember that GW frequency is 2X the orbital frequency
               'MaxTDelay': 14000, #units are Myr
               'DeltaTGalMyr': 50, #Time step resolution in the Galactic SFR
               #Extra options
               'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualizations
               'OneBinToUse': 10, #Number of the bin, if only one bin in used
               'NPoints': 1e5 # Number of stars to sample if we just sample present-day stars
    }

    #Find the norm for the different Galactic components, and define the weights
    #Store it in # NormCSet - the normalisation constant array
    try:
        NormConstantsDF   = pd.read_csv('./Data/BesanconGalacticConstants.csv')
        NormConstantsDict = NormConstantsDF.to_dict(orient='list')
    else:
        NormCSet = np.zeros(len(BesanconParamsDefined['BinName']),dtype=float)
        
        # Calculate volume integral for each Galactic component
        IntList = []
        for ii in range(10):
            Int  = GetVolumeIntegral(ii)
            IntList.append(Int)

        #Next population NormCSet for each component 
        
        #The halo and bulge masses have been independently determined so treat them first
        #The masses are predefined in GalaxyParams
        #Halo mass:
        IHalo            = np.where(BesanconParamsDefined['BinName'] == 'Halo')[0][0]
        NormCSet[IHalo]  = GalaxyParams['MHalo'] / IntList[IHalo]
        #Bulge mass:
        IBulge           = np.where(BesanconParamsDefined['BinName'] == 'Bulge')[0][0]
        NormCSet[IBulge] = (GalaxyParams['MBulge'] + GalaxyParams['MBulge2']) / IntList[IBulge]

        #The thin and thick disk masses are determined by the solar density and total Galaxy mass
        #which are predefined in GalaxyParams
        
        #Thin/Thick disc masses:
        #First, get the non-weighted local densities
        RhoTildeSet      = np.array([RhoBesancon(GalaxyParams['RGalSun'], GalaxyParams['ZGalSun'], ii) for ii in range(8)],dtype=float)        
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
        
        
    GalFunctionsDict = {'Besancon': RhoBesancon}


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


