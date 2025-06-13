#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %matplotlib inline

"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey, reinhold
"""

# +
import os, sys
import time
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from astropy import units as u
from astropy.constants import G, c, M_sun
from fractions import Fraction
import legwork as lw
import legwork.visualisation as vis

# Add local scripts
sys.path.insert(0, '..')
from galaxy_scripts.utils import MWConsts, fGW_from_A, calculateSeparationAfterSomeTime, inspiral_time, chirp_mass
from galaxy_scripts.galaxy_models.draw_samples_from_galaxy import createGalaxyModel
from galaxy_scripts.population_synthesis.get_popsynth_lisa_dwds import getLisaDwdProperties

# This environment variable must be set by the user to point to the root of the google drive folder
ROOT_DIR = os.environ['UCB_ROOT_DIR']
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']

# +
sys.path.append('/home/rwillcox/astro/ExternalRepos/fastgb/')
import lisaconstants as lc
import fastgb.fastgb as fb

from lisaorbits import EqualArmlengthOrbits, KeplerianOrbits
# -

# Units are MSun, kpc, Gyr
# FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER



# +
def matchDwdsToGalacticPositions(
        pathToPopSynthData=None,
        pathToGalacticSamples=None,
        applyInitialLisaBandFilter=True,
        num_desired_legwork_rows = 1e6,
        num_rng_samples = 1e7,
        num_galaxy_samples=2e7,
        randomSeed = None,
    ):

    # Initializers
    time0 = time.time()
    time_abs_beginning = time0
    rng = np.random.default_rng(randomSeed)
    
    # Convert large number inputs to ints
    num_desired_legwork_rows = int(num_desired_legwork_rows)
    num_rng_samples = int(num_rng_samples)
    num_galaxy_samples = int(num_galaxy_samples)

    # Import Population Synthesis data 
    massFraction = 1e6
    DWDs_from_popsynth, Z = getLisaDwdProperties(pathToPopSynthData, applyInitialLisaBandFilter=applyInitialLisaBandFilter)
    M1, M2, A_birth, T_DWD = DWDs_from_popsynth
    print()
    print("Grabbing popsynth data:")
    print(time.time() - time0)
    time0 = time.time()
    #Msun Msun Rsun     Myr 
    nBinaries_from_popsynth  = M1.shape[0]
    print(nBinaries_from_popsynth)

    # Import Galactic position samples
    galaxyModel = createGalaxyModel('Besancon', Z, saveOutput=False) 
    print()
    print("Create Galaxy Model:")
    print(time.time() - time0)
    time0 = time.time()
    drawn_samples = galaxyModel.DrawSamples(num_galaxy_samples)
    B_gal, L_gal, D_gal, T_gal_birth, _ = drawn_samples #[:,:nBinaries]
    print()
    print("Draw from Galaxy Model:")
    print(time.time() - time0)
    time0 = time.time()

    # WD mass-radius relation
    WD_MR_relation = UnivariateSpline(*np.loadtxt(os.path.join(ROOT_DIR, 'population_synthesis/WDMassRadiusRelation.dat')).T, k=4, s=0)
    print()
    R1 = WD_MR_relation(M1) # R_sun 
    R2 = WD_MR_relation(M2) # R_sun
    print("WD-MR relation:")
    print(time.time() - time0)
    time0 = time.time()

    print(done)
    #### This is the part that combines them! It loops so optimize it...
    legwork_input_matrix = np.zeros((num_desired_legwork_rows, 5)) # m1, m2, forb_tod, dist, ecc
    num_filled_rows = 0
    time_loop_0 = time.time()
    while num_filled_rows < num_desired_legwork_rows:
        
        time0 = time.time()
        
        # Reshuffle all of the galactic positions so that the position - binary matches are unique 
        idx_popsynth = rng.integers(low=0, size=num_rng_samples, high=nBinaries_from_popsynth)
        idx_galaxy   = rng.integers(low=0, size=num_rng_samples, high=num_galaxy_samples)
        
        m1 = M1[idx_popsynth]                  
        m2 = M2[idx_popsynth]    
        a_birth = A_birth[idx_popsynth]              
        t_dwd = T_DWD[idx_popsynth]           
        r1 = R1[idx_popsynth]                  
        r2 = R2[idx_popsynth]    

        t_birth = T_gal_birth[idx_galaxy]
        d_gal = D_gal[idx_galaxy]
        
        # Calculate present day properties of WDs 
        dt = t_birth - t_dwd # Myr
        a_today = calculateSeparationAfterSomeTime(m1, m2, a_birth, dt) # R_sun
        fGW_today = fGW_from_A(m1, m2, a_today) # Hz
        forb_today = fGW_today/2

        # Mask for non-mergers and those with valid GW freqs, apply the mask to the DWDs
        mask_mergers = a_today < r1+r2
        mask_lisa_band = (fGW_today > 1e-4) & (fGW_today < 1e-1) 
        mask = ~mask_mergers & mask_lisa_band

        num_new_rows = np.sum(mask)
        new_rows = np.vstack((m1, m2, forb_today, d_gal))[:,mask]

        # Cap the number of new rows if at the end of the matrix
        if num_filled_rows + num_new_rows > num_desired_legwork_rows:
            num_new_rows = num_desired_legwork_rows - num_filled_rows
            new_rows = new_rows[:,:num_new_rows]
                
        legwork_input_matrix[num_filled_rows:num_filled_rows + num_new_rows,:4] = new_rows.T
        num_filled_rows += num_new_rows
        
        print()
        print("Num new rows = ", num_new_rows)
        print("Num filled rows = ", num_filled_rows)
        
    print()
    print("All loops:")
    print(time.time() - time_loop_0)
    time0 = time.time()
    
    # Legwork 
    time0 = time.time()
    #m1, m2, a_today, fGW_today, b_gal, l_gal, d_gal = DWDs
    sources = lw.source.Source(
        m_1   = legwork_input_matrix[:,0] *u.Msun, 
        m_2   = legwork_input_matrix[:,1] *u.Msun, 
        f_orb = legwork_input_matrix[:,2] *u.Hz,
        dist  = legwork_input_matrix[:,3] *u.kpc, 
        ecc   = legwork_input_matrix[:,4]
        )
    print()
    print("Legwork sources:")
    print(time.time() - time0)
    time0 = time.time()
    
    snr = sources.get_snr(verbose=False)
    print()
    print("Legwork snr:")
    print(time.time() - time0)
    time0 = time.time()
    
    print("Total run time everything = {}".format(time.time() - time_abs_beginning))
            
    return sources
    
sources = matchDwdsToGalacticPositions( pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"), #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
             applyInitialLisaBandFilter=False)

# +
# Add l and b to the output
# And the ID of the binary...
# This is the GW dataset...


# +
# Plot Legwork 

time0 = time.time()

snr = sources.get_snr(verbose=False)
print("Legwork snr:")
print(time.time() - time0)
time0 = time.time()

# Visuals
cutoff = 1
#fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="kde", log_scale=(True, True), fill=True, xlim=(2e-6, 2e-1), which_sources=snr>cutoff, show=False)
fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="scatter", log_scale=(True, True), xlim=(2e-6, 2e-1), which_sources=snr>cutoff, show=False)
ax.set_ylim(1e-7, 1e6)
right_ax = ax.twinx()
freq_range = np.logspace(np.log10(2e-6), np.log10(2e-1), 1000)*u.Hz
vis.plot_sensitivity_curve(frequency_range=freq_range, fig=fig, ax=right_ax)

print("Legwork visuals:")
print(time.time() - time0)
# -






# +

fdot_prefactor = 96/5*np.power(np.pi,8/3)
fdot_prefactor #* np.power(1*u.Hz, 11/3)

# +
# Local, Jonathan Menu version
m1, m2, a_today, fGW_today, b_gal, l_gal, d_gal = DWDs

strain_prefactor = (4*np.power(G, 5/3)*np.power(np.pi, 2/3) / np.power(c, 4)).to((u.kpc  * u.Hz ** Fraction(-2/3)) / (u.Msun ** Fraction(5/3))).value # kpc Hz^(-2/3) / Msun^(5/3)
def get_strain_amplitude(Mc, dist, fGW):
    # Mc in Msun
    # dist in kpc
    # fGW in Hz
    #amp = 2*np.power(lc.GRAVITATIONAL_CONSTANT*Mc, 5/3)/(np.power(c,4)*dist)*np.power(np.pi*fGW,2/3)
    amp = strain_prefactor *np.power(Mc, 5/3)/(dist*np.power(fGW,2/3))
    return amp

def get_fdot(Mc, fGW):
    #fdot = 96./5.*np.power(np.pi,8/3)*np.power(lc.GRAVITATIONAL_CONSTANT*Mc/np.power(c,3),5/3)*np.power(fGW,11/3)
    fdot = fdot_prefactor * np.power(Mc,5/3)*np.power(fGW,11/3)
    return fdot


theta = np.pi/2 - params_cat[:,3]
phi = params_cat[:,4]
psi = params_cat[:,5]
cosiota = np.cos(params_cat[:,6])
A = params_cat[:,2]

# phi = ecliptic lambda
# theta = pi/2 - ecliptic beta
# psi is the polarization angle, [0,2pi]
# cosiota is the inclination angle, sample (in cos) randomly from -1 to 1?  need to check

# The Dp/Dc terms depend only on theta and phi, the sky positions, and can be computed from the galaxy samples
cTheta = np.cos(theta)
sTheta = np.sin(theta)
cPhi = np.cos(phi)
sPhi = np.sin(phi)
s2Phi = np.sin(2*phi)
cTheta2 = cTheta*cTheta
sTheta2 = sTheta*sTheta
cTheta4 = cTheta2*cTheta2
cPhi2 = cPhi*cPhi
s2Phi2 = s2Phi*s2Phi
DpDc = 243/512*cTheta * s2phi*(2*cPhi2-1)*(1+cTheta2)
Dc2 = 3/512*(120*sTheta2  + cTheta2 + 162*s2Phi2*cTheta2)
Dp2 = 3/2048*(487 + 158*cTheta2 + 7*cTheta4 - 162*s2Phi2*np.power(1 + cTheta2, 2))

# The Fp/Fc terms and A depend on the D's, but then only on the inclination and the polarization angle
# Can sample these randomly and attach them to the galactic positions as well, for ease of computation
c2Psi2 = np.power(np.cos(2*psi), 2)
s2Psi2 = 1 - c2Psi2
s4Psi = np.sin(4*psi)
cIota2 = cIota*cIota
Fp2 = (c2Psi2*Dp2 - s4Psi*DpDc + s2Psi2*Dc2)/4
Fc2 = (c2Psi2*Dc2 + s4Psi*DpDc + s2Psi2*Dp2)/4
sky_avg = np.sqrt((np.power(1+cIota2,2)*Fp2 + 4*cIota2*Fc2)/2) # the part that doesn't depend on A
strain = A * sky_avg



#Fp2a = (c2Psi2*Dc2 - s4Psi*DpDc + s2Psi2*Dp2)/4
#Fp2b = 1./4.*(np.cos(2*psi)**2*Dp2 - np.sin(4*psi)*DpDc + np.sin(2*psi)**2*Dc2)
# -

len(m1)
# upper limit is ~10M points




# +

#if __name__ == "__main__":
#    #DWDs = matchDwdsToGalacticPositions( pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
#    #                                     #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
#    #                                     applyInitialLisaBandFilter=False)
#    
#    maskApplyLisaMasks(DWDs,  useGenericLisaBandMask=False, useLegworkMask=True,  )
# -



# +
#import legwork.visualisation as vis

#def maskApplyLisaMasks(
#    DWDs=None, 
#    useGenericLisaBandMask=False,
#    useLegworkMask=False, # If true, calculate LISA visibile sources according to Legwork                                                                                                   
#    useFastGBMask=False,
#):
# ensure that only one mask has been chosen!

# Extract data

    #ra=10.68458*u.degree, dec=41.26917*u.degree, frame='icrs')
#print(DWDs)


nDWDs = m1.size

print(nDWDs)

#createLegWorkMask(m1, m2, d_gal, fGW_today, nDWDs)
sources = lw.source.Source(
    m_1   = m1 *u.Msun, 
    m_2   = m2 *u.Msun, 
    dist  = d_gal *u.kpc, 
    f_orb = fGW_today/2 *u.Hz,
    ecc   = np.zeros_like(m1)
    )






#n_values = 1500
#m_1 = np.random.uniform(0, 10, n_values) * u.Msun
#m_2 = np.random.uniform(0, 10, n_values) * u.Msun
#dist = np.random.normal(8, 1.5, n_values) * u.kpc
#f_orb = 10**(-5 * np.random.power(3, n_values)) * u.Hz
#ecc = 1 - np.random.power(5, n_values)
#sources = lw.source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb)



#print(np.sort(snr))
#mask_detectable_sources = sources.snr > cutoff
##print(np.sum(mask_detectable_sources))
#mask = mask_detectable_sources
#return fig
#return mask
##if useGenericLisaBandMask:
##    mask = createGenericLisaBandMask(fGW_today)
##if useLegworkMask:
##    mask = createLegWorkMask(m1, m2, d_gal, fGW_today, nDWDs)
##if useFastGBMask:
##    mask = createFastGBMask(...) #m1, m2, d_gal, fGW_today, nDWDs)

#fig
#masked_dwds = dwd_properties[:,mask]
#masks = [mask, mask_lisa_band]
#return DWDs[:,mask]


#if __name__ == "__main__":
#    #DWDs = matchDwdsToGalacticPositions( pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
#    #                                     #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
#    #                                     applyInitialLisaBandFilter=False)
#    
#    maskApplyLisaMasks(DWDs,  useGenericLisaBandMask=False, useLegworkMask=True,  )
# +

cutoff = 1
fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="kde", log_scale=(True, True), fill=True, xlim=(2e-6, 2e-1), which_sources=snr>cutoff, show=False)
ax.set_ylim(1e-7, 1e6)
right_ax = ax.twinx()
freq_range = np.logspace(np.log10(2e-6), np.log10(2e-1), 1000)*u.Hz
vis.plot_sensitivity_curve(frequency_range=freq_range, fig=fig, ax=right_ax)

# +

cutoff = 0.0 #1
fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="scatter", log_scale=(True, True), xlim=(2e-6, 2e-1), which_sources=snr>cutoff, show=False)
ax.set_ylim(1e-7, 1e6)
right_ax = ax.twinx()
freq_range = np.logspace(np.log10(2e-6), np.log10(2e-1), 1000)*u.Hz
vis.plot_sensitivity_curve(frequency_range=freq_range, fig=fig, ax=right_ax)
# -







# +
def createGenericLisaBandMask(
        fGW_today,
        f_min = 1e-4, # Minimum frequency bin for binary to reach LISA within Hubble time
        f_max = 1e-1, # Maximum frequency bin for LISA 
    ):
    # Remove systems that would have merged by now 
    mask_lisa_band = (fGW_today > f_min) & (fGW_today < f_max)
    mask = ~mask_mergers #& mask_lisa_band
    return mask

def createLegWorkMask(
    m1,
    m2,
    d_gal,
    fGW_today,
    nDWDs,
    cutoff=2, # user should set this
    ):
    sources = lw.source.Source(
        m_1   = m1 *u.Msun, 
        m_2   = m2 *u.Msun, 
        dist  = d_gal *u.kpc, 
        f_orb = fGW_today/2 *u.Hz,
        ecc   = np.zeros_like(nDWDs),
        )
    
    snr = sources.get_snr(verbose=True)
    print(np.sort(snr))
    mask_detectable_sources = sources.snr > cutoff
    print(np.sum(mask_detectable_sources))
    mask = mask_detectable_sources
    return mask


# +
# TODO: check units!!
# TODO: pull out constants, calculate these once at the top!

# Questions: what units are expected for the params_cat inputs?

c = 3e8
def get_strain_amplitude(Mc, dist, fGW):
    Ld = dist*1e3*lc.PARSEC
    amp = 2*np.power(lc.GRAVITATIONAL_CONSTANT*Mc, 5/3)/(np.power(c,4)*dist)*np.power(np.pi*fGW,2/3)
    return amp

def get_fdot(Mc, fGW):
    fdot = 96./5.*np.power(np.pi,8/3)*np.power(lc.GRAVITATIONAL_CONSTANT*Mc/np.power(c,3),5/3)*np.power(fGW,11/3)
    return fdot

def createFastGBMask(DWDs,
    #m1, m2, d_gal, fGW_today, nDWDs
    ):
    #gb = GBGPU(use_gpu=False)
    
    m1, m2, a_today, fGW_today, b_gal, l_gal, d_gal = DWDs
    nDWDs = m1.size
    print(nDWDs)
    num_bin = nDWDs
    #dt = 10.0
    #Tobs = 1.0 * YEAR
    t0 = 0 # s
    dt = 10 # s
    
    Tobs = 62914560/2 # s
    Twindow = 0 # extension factor for windowing: [t0-Twindow*Tobs, t0 + Tobs + Twindow*Tobs]
    
    print("Simulated:", Tobs/lc.ASTRONOMICAL_YEAR, "years")
    
    dt_orbits = 10000
    size_orbits = int(np.ceil((t0 + Tobs + Twindow * Tobs + 10 - (t0 - Twindow*Tobs - 10)) / dt_orbits))
    #orbits = KeplerianOrbits()
    #orbits.write("./orbits.h5", t0=t0 - 2 * Tobs - 10, dt=dt_orbits, size=size_orbits)

    #amp = 2e-23  # amplitude
    #f0 = 2e-3  # f0
    #fdot = 7.538331e-18  # fdot
    fddot = 0.0 # fddot
    phi0 = 0.1  # initial phase
    iota = 0.2  # inclination
    psi = 0.3  # polarization angle
    lam = 0.4  # ecliptic longitude
    beta_sky = 0.5  # ecliptic latitude
    
    # for batching
    #amp_in = np.full(num_bin, amp)
    #f0_in = np.full(num_bin, f0)
    #fdot_in = np.full(num_bin, fdot)
    #fddot_in = np.full(num_bin, fddot)
    phi0_in = np.full(num_bin, phi0)
    iota_in = np.full(num_bin, iota)
    psi_in = np.full(num_bin, psi)
    #lam_in = np.full(num_bin, lam)
    #beta_sky_in = np.full(num_bin, beta_sky)

    #m1 = .4 # mass m1 in Msun
    #m2 = .5 # mass m2 in Msun
    #Ld = 1 # luminosity distance in kpc
    #M = m1 + m2 # total mass in Msun
    #nu = m1 * m2 / M**2 # mass ratio (dimensionless)
    #Mc = nu**(3./5.)*(m1+m2)*lc.SUN_MASS
    
    Mc = chirp_mass(m1, m2)*lc.SUN_MASS
    amp = get_strain_amplitude(Mc, d_gal, fGW_today)
    fdot = get_fdot(Mc, fGW_today)

    # TODO: !!! b_gal and l_gal are not the same as ecliptic lat and long. Need a long term solution that avoids lots of converting !!!
    params_cat = np.array([
        fGW_today, #fid["Frequency"][:],
        fdot,    #fid["FrequencyDerivative"][:],
        amp,    #fid["Amplitude"][:],
        b_gal,    #fid["EclipticLatitude"][:],
        l_gal,    #fid["EclipticLongitude"][:],
        psi_in,    #fid["PolarisationAngle"][:],
        iota_in,    #fid["InclinationAngle"][:],
        phi0_in,    #fid["InitialPhase"][:]
    ])
    
    params_cat = params_cat.T
    #print("Highest frequency (Hz):", np.max(params_cat[:,0]))

    # Crucial pieces!!
    myfgb = fb.FastGB(delta_t=dt, T=Tobs, N=1024)#, orbits="orbits.h5")# , Twindow=Twindow, advanced=True)
    Xf, Yf, Zf, kmin = myfgb.get_fd_tdixyz(params_cat) # Frequency domain response
    
    f = np.arange(0, 1/dt, 1/((2*Twindow+1)*Tobs))
    totalresponseXYZf = np.zeros((3,len(f)), dtype=complex)
    source_limit = 1000 #!!
    for i in range(source_limit):
        totalresponseXYZf[0, kmin[i]:(kmin[i]+len(Xf[i]))] += Xf[i] # TDI X
        totalresponseXYZf[1, kmin[i]:(kmin[i]+len(Yf[i]))] += Yf[i] # TDI Y
        totalresponseXYZf[2, kmin[i]:(kmin[i]+len(Zf[i]))] += Zf[i] # TDI Z
        
    #plt.loglog(f, abs(totalresponseXYZf[0])) # TDI X
createFastGBMask(DWDs) #m1, m2, d_gal, fGW_today, nDWDs)
# -
myf = h5.File('orbits.h5', 'r')
myf.keys()











# +
l, b = l_gal[0], b_gal[0]
coord = Galactic(Angle(l, unit=u.rad), Angle(b, unit=u.rad)).transform_to(BarycentricTrueEcliptic())

coord

# +
from lisaconstants import c, PARSEC, SUN_MASS, GRAVITATIONAL_CONSTANT

from astropy.coordinates import Galactic, Angle, BarycentricTrueEcliptic
from astropy import units as u

def conv_lb_lonlat(l,b):
    coord = Galactic(Angle(l, unit=u.rad), Angle(b, unit=u.rad)).transform_to(BarycentricTrueEcliptic())
    return coord.lat.rad, coord.lon.rad # ecl_lon, ecl_lat in rad
    
m1, m2, a_today, fGW_today, b_gal, l_gal, d_gal = DWDs
ecl_lat, ecl_lon = conv_lb_lonlat(l_gal, b_gal)
nDWDs = m1.size
print(nDWDs)
    
#m1 = .4 # mass m1 in Msun
#m2 = .5 # mass m2 in Msun
#Ld = 1 # luminosity distance in kpc
#M = m1 + m2 # total mass in Msun
#nu = m1 * m2 / M**2 # mass ratio (dimensionless)
#Ld = Ld*1e3*PARSEC
c4 = np.power(c,4)  # L^4 / T^4
c5 = np.power(c,5)  # L^4 / T^4
f0 = fGW_today

def get_nu(m1, m2):
    nu = m1 * m2 / np.power(m1+m2,2) # mass ratio (dimensionless)
    return nu
    
def get_chirp_mass(m1, m2):
    nu = get_nu(m1, m2)
    Mc = np.power(nu, 3./5.)*(m1+m2)*SUN_MASS
    return Mc
Mc = get_chirp_mass(m1, m2)

def get_gw_amplitude(f0, Mc, d_gal):
    amp =  2*np.power(np.pi*f0, 2/3)*np.power(GRAVITATIONAL_CONSTANT*Mc, 5/3) / (c4*d_gal)
    return amp
amp = get_gw_amplitude(f0, Mc, d_gal)

fdot_prefactor = 96./5.*np.power(np.pi, 8/3) # dimensionless
def get_fdot(Mc, f0):
    fdot = fdot_prefactor *np.power(GRAVITATIONAL_CONSTANT*Mc, 5/3) *np.power(f0,11/3)/ c5
    return fdot
fdot = get_fdot(Mc, f0)

t0 = 0 # s
dt = 10 # s

Tobs = 62914560/2 # s
Twindow = 0 # extension factor for windowing: [t0-Twindow*Tobs, t0 + Tobs + Twindow*Tobs]

print("Simulated:", Tobs/constants.ASTRONOMICAL_YEAR, "years")

dt_orbits = 10000
size_orbits = int(np.ceil((t0 + Tobs + Twindow * Tobs + 10 - (t0 - Twindow*Tobs - 10)) / dt_orbits))
print(size_orbits)
orbits = KeplerianOrbits()
orbits.write("./orbits.h5", t0=t0 - 2 * Tobs - 10, dt=dt_orbits, size=size_orbits)

# +
# Sample angles randomly
polarizationAngle   = np.random.rand(nDWDs)  *2*np.pi - np.pi # [-pi,pi]          
cosInclinationAngle =  np.random.rand(nDWDs) *2 - 1        # [-1, 1]   
initialPhase        = np.random.rand(nDWDs)  *2*np.pi         # [0, 2pi]

params_cat = np.array([
        f0, #fid["Frequency"][:],
        fdot, #fid["FrequencyDerivative"][:],
        amp, #fid["Amplitude"][:],
        ecl_lat, #fid["EclipticLatitude"][:],
        ecl_lon, #fid["EclipticLongitude"][:],
        polarizationAngle  ,#fid["PolarisationAngle"][:],
        cosInclinationAngle,#fid["InclinationAngle"][:],
        initialPhase       ,#fid["InitialPhase"][:]
])

params_cat = params_cat.T
print("Highest frequency (Hz):", np.max(params_cat[:,0]))
params_cat[:,0]
# -



# +
myfgb = fb.FastGB(delta_t=dt, T=Tobs, N=1024, orbits="orbits.h5")# , Twindow=Twindow, advanced=True)

Xf, Yf, Zf, kmin = myfgb.get_fd_tdixyz(params_cat)

f = np.arange(0, 1/dt, 1/((2*Twindow+1)*Tobs))
totalresponseXYZf = np.zeros((3,len(f)), dtype=complex)
source_limit = 1000 #!!
for i in range(source_limit):
    totalresponseXYZf[0, kmin[i]:(kmin[i]+len(Xf[i]))] += Xf[i] # TDI X
    totalresponseXYZf[1, kmin[i]:(kmin[i]+len(Yf[i]))] += Yf[i] # TDI Y
    totalresponseXYZf[2, kmin[i]:(kmin[i]+len(Zf[i]))] += Zf[i] # TDI Z
    
plt.loglog(f, abs(totalresponseXYZf[0])) # TDI X
# -





# +
o for the SNR calculation, I relied on https://gitlab.in2p3.fr/Nikos/gwg/-/blob/master/gwg/gwg.py to make a quick SNR script: from ldc.lisa.noise.noise import AnalyticNoise
fpersource = np.array([f[kmin[i]:(kmin[i]+len(Xf[i]))] for i in range(source_limit)])
def XYZ2AET(X, Y, Z):
    return np.array([(Z - X)/np.sqrt(2.0),
            (X - 2.0*Y + Z)/np.sqrt(6.0),
            (X + Y + Z)/np.sqrt(3.0)])
channels = ["A", "E", "T"]
S = np.zeros( (3, fpersource.shape[0], fpersource.shape[1]) )
for i, C in enumerate(channels):
    S[i] = np.array( AnalyticNoise(fpersource).psd(option=C, freq=fpersource, tdi2=False) )
D = XYZ2AET(Xf,Yf,Zf)
inner_prod = np.real(D*np.conj(D) / np.array(S))
snr = np.sum( np.sum(inner_prod, axis=-1), axis=0 )* 4.0*abs(fpersource[:,1]-fpersource[:,0])
GitLabGitLab
gwg/gwg.py · master · Nikos Karnesis / gwg · GitLab
Code for estimating the confusion noise signal from GWs emitted from the compact Galactic Binaries, as measured by LISA.




NEW

4:23
goes really fast
4:24
~1 ms per source: your integral only covers a very small domain
4:24
(i.e., the "peak")
4:24
regarding the coordinate transformation: I'd use barycentric
