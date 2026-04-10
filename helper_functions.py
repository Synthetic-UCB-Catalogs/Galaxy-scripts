import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import UnivariateSpline
import os

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

CodeDir       = os.path.dirname(os.path.abspath(__file__))
MRWDMset,MRWDRSet    = np.split(np.loadtxt(CodeDir + '/WDData/MRRel.dat'),2,axis=1)
MRWDMset             = MRWDMset.flatten()
MRWDRSet             = MRWDRSet.flatten()
MRSpl                = UnivariateSpline(MRWDMset, MRWDRSet, k=4, s=0)

def RWD(MWD):
    """
    WD radius in RSun
    """
    Res = MRSpl(MWD)
    return Res

def POrbYr(MDonor, MAccretor, BinARSun):
    """
    The orbital period in years
    """
    Omega = np.sqrt(GNewtCGS*MSunToG*(MDonor + MAccretor)/(BinARSun*RSunToCm)**3)
    Res   = (2.*np.pi/Omega)/YearToSec
    return Res

def ABinRSun(MDonor, MAccretor, POrbYr):
    """
    The binary separation in RSun
    """
    Omega = (2.*np.pi/(POrbYr*YearToSec))
    Res = ((GNewtCGS*MSunToG*(MDonor + MAccretor)/Omega**2)**(1/3))/RSunToCm
    return Res

def TGWMyr(M1MSun, M2MSun, aRSun):
    """
    The GW inspiral time in megayears
    """
    Res = TauGWPreFacMyr/((M1MSun + M2MSun)*M1MSun*M2MSun/aRSun**4)
    return Res

def APostGWRSun(M1MSun, M2MSun, AInitRSun, TGWInspMyr):
    """
    The orbital separation after a given GW inspiral time
    """
    TGWFull = TGWMyr(M1MSun, M2MSun, AInitRSun)
    Res     = AInitRSun*(1 - TGWInspMyr/TGWFull)**0.25
    return Res

def fRL(q):
    """
    Roche lobe radius/BinA -- Eggleton's formula
    """
    X   = q**(1./3)
    Res = 0.49*(X**2)/(0.6*(X**2) + np.log(1.+X))
    return Res

def fRLDonor(MDonorMSun,MAccretorMSun):
    """
    Roche lobe radius/BinA for the donor
    """
    q = MDonorMSun/MAccretorMSun
    return fRL(q)

def get_galactic_coords(XRel,YRel,ZRel):
    """
    Get Galactic l and b coordinates from the relative X,Y,Z coordinates.
    """
    RRel = np.empty(len(XRel), dtype=np.float64)
    np.square(XRel, out=RRel)
    RRel += YRel * YRel
    RRel += ZRel * ZRel
    np.sqrt(RRel, out=RRel)
    
    Galb = np.arcsin(ZRel/RRel)
    Gall = np.mod(np.arctan2(YRel, XRel), 2.*np.pi)
    return Gall, Galb

def get_f_gw_from_semimajor(m1, m2, a):
    """
    Compute orbital frequency in Hz.

    Parameters
    ----------
    a : float or array
        Semi-major axis in solar radii.
    m1, m2 : float or array
        Masses in solar masses.

    Returns
    -------
    f_orb : float or array
        Orbital frequency in Hz.
    """
    G = 6.67430e-11          # m^3 kg^-1 s^-2
    M_sun = 1.98847e30       # kg
    R_sun = 6.957e8          # m.
    a_m = a * R_sun
    m_total = (m1 + m2) * M_sun

    f_orb = (1 / (2 * np.pi)) * np.sqrt(G * m_total / a_m**3)
    return 2*f_orb
