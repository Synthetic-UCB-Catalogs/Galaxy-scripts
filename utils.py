from astropy import units as u
from astropy import constants as const
import numpy as np

# Common astrophysical functions and constants
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



# RTW: Do we need both? 
#MW parameters
use_alt_params=False
if not use_alt_params:
    MWConsts = {'MGal': 6.43e10, #From Licquia and Newman 2015
                'MBulge1': 6.1e9, #From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8, #From Robin+ 2012, metal-poor bulge
                'MHalo': 1.4e9, #From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'RGalSun': 8.2, #Bland-Hawthorn, Gerhard 2016
                'ZGalSun': 0.025 #Bland-Hawthorn, Gerhard 2016
               }
else:
    MWConsts = {'MGal': 6.43e10, #From Licquia and Newman 2015
                'MBulge1': 6.1e9, #From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8, #From Robin+ 2012, metal-poor bulge
                'MHalo': 1.4e9, #From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'RGalSun': 8.122, #GRAVITY Collaboration et al. 2018
                'ZGalSun': 0.028 #Bennett & Bovy, 2019
               }
MWConsts.update({'MBulge': MWConsts['MBulge1'] + MWConsts['MBulge2']})
#3D components of the Sun’s velocity (U ; V ;W ) =(12:9; 245:6; 7:78) km s^1 (Drimmel & Poggio 2018)




#GW constants
ADotGWPreFacCGS = (256./5)*((CLightCGS)**(-2))*(GNewtCGS*MSunToG/(RSunToCm*CLightCGS))**3
TauGWPreFacMyr  = (RSunToCm/ADotGWPreFacCGS)/MyrToSec

#WD radius in RSun
def RWDPre(MWD):
    Res = float(MRSpl(MWD))
    return Res

#The orbital period in years
def POrbYrPre(MDonor, MAccretor, ComponentARSun):
    Omega = np.sqrt(GNewtCGS*MSunToG*(MDonor + MAccretor)/(ComponentARSun*RSunToCm)**3)
    Res   = (2.*np.pi/Omega)/YearToSec
    return Res

#The binary separation in RSun
def AComponentRSunPre(MDonor, MAccretor, POrbYr):
    Omega = (2.*np.pi/(POrbYr*YearToSec))
    Res = ((GNewtCGS*MSunToG*(MDonor + MAccretor)/Omega**2)**(1/3))/RSunToCm
    return Res


#The GW inspiral time in megayears
def TGWMyrPre(M1MSun, M2MSun, aRSun):
    Res = TauGWPreFacMyr/((M1MSun + M2MSun)*M1MSun*M2MSun/aRSun**4)
    return Res

#The orbital separation after a given GW inspiral time
def APostGWRSunPre(M1MSun, M2MSun, AInitRSun, TGWInspMyr):
    TGWFull = TGWMyr(M1MSun, M2MSun, AInitRSun)
    Res     = AInitRSun*(1 - TGWInspMyr/TGWFull)**0.25
    return Res

#Roche lobe radius/ComponentA -- Eggeleton's formula
def fRL(q):
    X   = q**(1./3)
    Res = 0.49*(X**2)/(0.6*(X**2) + np.log(1.+X))
    return Res

#Roche lobe radius/ComponentA for the donor
def fRLDonor(MDonorMSun,MAccretorMSun):
    q = MDonorMSun/MAccretorMSun
    return fRL(q)



RWD = np.vectorize(RWDPre)


POrbYr = np.vectorize(POrbYrPre)

AComponentRSun = np.vectorize(AComponentRSunPre)


TGWMyr = np.vectorize(TGWMyrPre)

APostGWRSun = np.vectorize(APostGWRSunPre)



### Volume integrators


#2D Volume integrator for the Galactic density components
def GetVolumeIntegral(XMax, YMax, ZMax, RhoFunction, NPoints=1000):
    
    # RTW: what units?
    #RMax = np.sqrt((GalaxyModelParams['XMax'][iComponent])**2 + (GalaxyModelParams['YMax'][iComponent])**2)
    #ZMax = GalaxyModelParams['ZMax'][iComponent]
    RMax = np.sqrt(XMax**2 + YMax**2)
    
    RArray = np.linspace(0, RMax, NPoints)
    ZArray = np.linspace(-ZMax, ZMax, NPoints)
    
    dR = RMax / (NPoints - 1)
    dZ = 2 * ZMax / (NPoints - 1)
  
    RGrid, ZGrid = np.meshgrid(RArray, ZArray, indexing='ij')
    #RhoGrid = GalaxyModelParams['RhoFunction'](RGrid, ZGrid, iComponent)
    RhoGrid = RhoFunction(RGrid, ZGrid, iComponent)
    
    TotalMass = np.sum(RhoGrid*RGrid) * dR * dZ *2*np.pi
    return TotalMass

#3D version for later - RTW: is this needed?
# =============================================================================
## Volume integrator for the Galactic density components
# def GetVolumeIntegral(iComponent):
#     NPoints = 400
#     
#     XMax = GalaxyModelParams['XMax'][iComponent]
#     YMax = GalaxyModelParams['YMax'][iComponent]
#     ZMax = GalaxyModelParams['ZMax'][iComponent]
#     
#     
#     XSet = np.linspace(-XMax, XMax, NPoints)
#     YSet = np.linspace(-YMax, YMax, NPoints)
#     ZSet = np.linspace(-ZMax, ZMax, NPoints)
#     
#     dX = 2 * XMax / (NPoints - 1)
#     dY = 2 * YMax / (NPoints - 1)
#     dZ = 2 * ZMax / (NPoints - 1)
#  
#     def RhoFun(X, Y, Z):
#         return GalaxyModelParams['RhoFunction'](np.sqrt(X**2 + Y**2), Z, iComponent)
#     
#     X, Y, Z = np.meshgrid(XSet, YSet, ZSet, indexing='ij')
#     RhoSet = RhoFun(X, Y, Z)
#     
#     Res = np.sum(RhoSet) * dX * dY * dZ
#     
#     return Res
# =============================================================================
