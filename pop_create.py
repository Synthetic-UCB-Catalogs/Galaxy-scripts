import astropy.units as u
import numpy as np
import legwork as lw
import pandas as pd
from rapid_code_load_T0 import load_T0_data
import utils
import os

def get_R_WD(M_WD):
    '''Calculates the radius of a white dwarf given its mass using 
    a spline fit to the mass-radius relation.
    
    Parameters
    ----------
    M_WD : float or array-like
        Mass of the white dwarf in solar masses.

    Returns
    -------
    float or array-like
        Radius of the white dwarf in solar radii.
    '''
    from scipy.interpolate import UnivariateSpline
    import os

    codeDir = os.path.dirname(os.path.abspath(__file__))

    # Load the mass-radius data for white dwarfs
    M_WD_dat, R_WD_dat = np.split(np.loadtxt(codeDir + '/WDData/MRRel.dat'),2,axis=1)
    M_WD_dat = M_WD_dat.flatten()
    R_WD_dat = R_WD_dat.flatten()

    # Create a 4th order spline that goes through every data point
    mass_radius_relation = UnivariateSpline(M_WD_dat, R_WD_dat, k=4, s=0)
    R_WD = mass_radius_relation(M_WD)

    return R_WD


def frac_RL(q):
    '''Calculates the Roche lobe radius in units of the binary separation.
    Parameters
    ----------
    q : float or array-like
        Mass ratio of the donor to the accretor (MDonor/MAccretor).
    Returns
    -------
    f : float or array-like
        Roche lobe radius in units of the binary separation.
    '''
    X = q**(1./3)
    f = 0.49*(X**2)/(0.6*(X**2) + np.log(1.+X))
    return f


def get_T0_DWDs(ModelParams, T0_dat_path, verbose=False):
    '''Returns the T0 data for the DWDs based on the model parameters.
    
    Parameters
    ----------
    ModelParams : dict
        Dictionary containing model parameters including 'run_sub_type'.
    T0_dat_path : str
        Path to the T0 data file.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.

    Returns
    -------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for each DWD that formed
    '''
    #Load the T0 data
    # Note that the path should be the path to 'data_products'
    # folder in the Google Drive download; the header of the T0
    # data file is saved in _ that is returned by the load_T0_data
    # function
    T0_data, _ = load_T0_data(T0_dat_path, code=ModelParams['code'], metallicity=0.02)
    
    if verbose:
        print(f"Loaded T0 data with {len(T0_data)} entries for the {ModelParams['code']} {ModelParams['run_wave']} {ModelParams['run_sub_type']} run.")

    # Handle the DWD selection based on the code in ModelParams
    # and apply a semimajor axis cut as specified in ModelParams
    if ModelParams['code'] == 'ComBinE':
        T0_DWD =  T0_data.loc[
            (T0_data.type1 == 2) & (T0_data.type2 == 2) & 
            (T0_data.semiMajor > 0) & (T0_data.semiMajor < ModelParams['semiMajor_max'])
            ].groupby('ID', as_index=False).first()
    else:
        T0_DWD = T0_data.loc[
            (T0_data.type1.isin([21,22,23])) & (T0_data.type2.isin([21,22,23])) & 
            (T0_data.semiMajor > 0) & (T0_data.semiMajor < ModelParams['semiMajor_max'])
            ].groupby('ID', as_index=False).first() 
        
    if verbose: 
        print(f"Found {len(T0_DWD)} DWDs in T0 data with semimajor axis < {ModelParams['semiMajor_max']} RSun.")
            
    return T0_DWD


def get_a_RLO(T0_DWD):
    '''Calculates the semimajor axis at Roche overflow for each DWD in the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.

    Returns
    -------
    a_rlo : array-like
        Array containing the semimajor axis at Roche overflow for each DWD.
    '''
    
    # First determine the separation where the donor fills its Roche lobe
    # Calculate the radius of the least massive WD 
    # [this will be the donor at mass transfer]
    r_don = get_R_WD(np.minimum(T0_DWD['mass1'].values,T0_DWD['mass2'].values))

    # Calculate the mass ratio where q = don/acc
    q_rlo = np.minimum(T0_DWD['mass1'], T0_DWD['mass2']) / np.maximum(T0_DWD['mass1'], T0_DWD['mass2'])

    # Calculate the Roche lobe radius in RSun
    a_rlo = r_don / frac_RL(q_rlo)

    return a_rlo


def get_a_LISA(T0_DWD, f_LISA_low=1e-4): 
    '''Calculates the semimajor axis at the lower bound of LISA 
    frequency for each DWD in the T0 data.
    Parameters
    ---------- 
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.
        
    Returns
    -------
    a_LISA : array-like
        Array containing the semimajor axis at the lower bound of LISA frequency for each DWD.
    '''
    # Calculate the semimajor axis at the lower bound of LISA sensitivity frequency
    a_LISA = lw.utils.get_a_from_f_orb(
        f_orb=0.5*f_LISA_low * u.Hz, 
        m_1=T0_DWD.mass1.values * u.Msun, 
        m_2=T0_DWD.mass2.values * u.Msun
    ).to(u.Rsun).value
    
    return a_LISA


def get_GW_timescales(T0_DWD):
    '''Calculates the GW inspiral timescales for each DWD in the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.

    Returns
    -------
    t_to_LISA : array-like
        Array containing the time to get to the LISA band from formation for each DWD.
    '''
    # Calculate the time to merger from DWD formation
    t_merge_gw = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['semiMajor'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calculate the time to merger from the Roche lobe overflow
    t_merge_rlo = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['a_rlo'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calculate the time to merger from the lower bound of the LISA band
    t_merge_LISA = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['a_LISA'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calcalate the time to get to the LISA band from formation
    t_to_LISA = np.zeros_like(t_merge_gw)
    # for DWDs that form in the LISA band, the time to get to the LISA band is 0

    # for DWDs that form with frequencies below the LISA band, the GW merger 
    # timescale is larger than the time to merge from the bottom of the LISA band, 
    # so the time to get to the LISA band is the difference between these two timescales
    below_LISA_band_mask = t_merge_gw > t_merge_LISA
    t_to_LISA[below_LISA_band_mask] = t_merge_gw[below_LISA_band_mask] - t_merge_LISA[below_LISA_band_mask]
    
    # add time to formation to get the absolute time to get to the LISA band
    t_to_LISA += T0_DWD['time'].values

    # Calculate the time to get to Roche lobe overflow from the formation of the DWD
    # this is just the difference between the time to merge from formation and the time to merge from Roche lobe overflow
    t_LISA_max = t_merge_gw - t_merge_rlo

    # add time to formation to get the absolute time to get to Roche lobe overflow
    t_LISA_max += T0_DWD['time'].values
    
    return t_to_LISA, t_LISA_max


def filter_for_potential_LISA_sources(T0_DWD, ModelParams, verbose=False):
    '''Calculates the properties of the DWDs based on the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.
    ModelParams : dict
        Dictionary containing model parameters including 'age_max'.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.

    Returns
    -------
    T0_DWD : DataFrame
        DataFrame with calculated properties for each DWD.
    '''

    # get the semimajor axis at Roche lobe overflow
    T0_DWD['a_rlo'] = get_a_RLO(T0_DWD)
    T0_DWD = T0_DWD.loc[T0_DWD['semiMajor'] > T0_DWD['a_rlo']].copy() # filter out DWDs that are already in Roche lobe overflow at T0

    # get the semimajor axis at the lower bound of LISA frequency
    T0_DWD['a_LISA'] = get_a_LISA(T0_DWD)

    # get the GW inspiral timescales
    T0_DWD['t_to_LISA'], T0_DWD['t_LISA_max'] = get_GW_timescales(T0_DWD)

    # filter based on sources that don't make it to LISA band before max age
    # since the t_to_LISA is the time to get to the LISA band from formation, 
    # we need to add the time of formation to get the absolute time to get to the LISA band
    T0_DWD = T0_DWD.loc[(T0_DWD['t_to_LISA'] < ModelParams['age_max'])]

    if verbose:
        print(f"Filtered DWDs to {len(T0_DWD)} that will evolve to LISA band before {ModelParams['age_max']} Myr.")

    return T0_DWD


def get_possible_T0_LISA_sources(ModelParams, T0_dat_path, verbose=False):
    '''Returns a list of likely LISA sources drawn from the 
    T0 data and the maximum Galactic component age 
    specified in ModelParams.
    
    Parameters
    ----------
    ModelParams : dict
        Dictionary containing model parameters including 'run_sub_type' and 'age_max'.
    T0_dat_path : str
        Path to the T0 data file.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.
    
    Returns
    -------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    '''

    # Get the T0 data for DWDs
    # Note that the path should be the path to 
    # 'data_products' directory in the Google Drive from rclone
    T0_DWD = get_T0_DWDs(ModelParams, T0_dat_path, verbose=verbose)
    if T0_DWD.empty:
        raise ValueError("No DWDs found in the T0 data. Check the input parameters or data file.")

    # Calculate orbital properties and GW evolution timescales based on the T0 WDs
    # This will also filter out DWDs that will not evolve to the LISA band before 
    # the maximum age specified in ModelParams['age_max']
    T0_DWD_LISA = filter_for_potential_LISA_sources(T0_DWD, ModelParams, verbose=verbose)
    if T0_DWD_LISA.empty:
        raise ValueError("No DWDs found that will evolve to the LISA band before the maximum age. Check the input parameters or T0 data file.")

    # clear the original T0_DWD DataFrame to save memory
    del T0_DWD

    return T0_DWD_LISA

def get_N_Gx_sample(T0_DWD_LISA, ModelParams):
    '''Returns the number of DWDs in the Galaxy based on the model parameters.

    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    ModelParams : dict
        Dictionary containing model parameters including 'run_sub_type'.
    
    Returns
    -------
    N_DWD_Gx : int
        Number of DWDs in the Galaxy.
    '''
    if ModelParams['run_wave'] == 'initial_condition_variations':
        mass_norm  = utils.get_mass_norm(IC_model=ModelParams['run_sub_type'], binary_fraction=0.5)
    elif ModelParams['run_wave'] == 'mass_transfer_variations':
        mass_norm  = utils.get_mass_norm(IC_model='fiducial', binary_fraction=0.5)
        #use fiducial mass norm for all mass transfer variations
    gx_to_sim_mass = utils.galaxy_params('MGal')/mass_norm
    
    if ModelParams['code'] == 'BPASS':
        #BPASS models have a different mass normalization, 
        #so we need to adjust the gx_to_sim_mass accordingly
        gx_to_sim_mass = gx_to_sim_mass * 20.9966

    N_DWD_Gx = len(T0_DWD_LISA) * gx_to_sim_mass
    N_DWD_Gx = int(N_DWD_Gx) + (np.random.uniform() < (N_DWD_Gx % 1))
    
    return N_DWD_Gx

def get_component_stats(gx_component, gx_component_df, ModelParams):
    """
    Compute sub-bin statistics for a given Galactic component.

    Parameters
    ----------
    gx_component : str
        Name of the component (e.g., 'ThinDisk1').
    gx_component_df : pd.DataFrame
        DataFrame containing the DWDs in this component.
        Must have 'AbsTimeToLISAMyr' and 'AbsTimeToLISAEndMyr'.
    ModelParams : dict
        Dictionary containing 'DeltaTGalMyr'.

    Returns
    -------
    stats : pd.DataFrame
        Columns: ['BinID', 'component_name', 'n_DWDs']
    """

    # Find the index of the Galactic component
    ii = utils.Besancon_params('BinName').tolist().index(gx_component)

    stats = pd.DataFrame({
        'BinID': [ii],
        'component_name': [gx_component],
        'n_DWDs': [len(gx_component_df)]
    })

    return stats

def sample_component_ages(gx_component, n_samp):
    '''Assigns ages to the DWDs in a given component based on the Besancon parameters.
    
    Parameters
    ----------
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    n_samp : int
        Number of DWDs in the Galaxy for this component.

    Returns
    -------
    ages : array-like
        Array of ages assigned to the DWDs in the component.
    '''
    ii = utils.Besancon_params('BinName').tolist().index(gx_component)
    if ii < 0:
        raise ValueError(f"Component {gx_component} not found in Besancon parameters.")
    
    # Get the age range and mass fraction for the component
    t_lo = utils.Besancon_params('AgeMin')[ii]
    t_hi = utils.Besancon_params('AgeMax')[ii]
    if np.all(t_lo == t_hi):
        ages = np.ones(n_samp) * t_lo
    else:
        ages = np.random.uniform(t_lo, t_hi, n_samp)
    return ages

def filter_possible_LISA_sources(gx_component_df, f_gw_LISA_low=1e-4, f_gw_LISA_high=1e-1):
    '''Filters the DWDs in a galaxy component DataFrame to those that are likely LISA sources.
    
    Parameters
    ----------
    gx_component_df : DataFrame
        DataFrame containing the DWDs in a specific galaxy component. 
    f_gw_LISA_low : float, optional (default=1e-4)
        Lower bound of the LISA frequency band in Hz.   
    f_gw_LISA_high : float, optional (default=1e-1)
        Upper bound of the LISA frequency band in Hz.


    Returns
    -------
    gx_component_df : DataFrame
        DataFrame containing only the DWDs that have a semimajor axis within the LISA band
    '''
    # cheap pre-filter: skip DWDs that can't be in the LISA band at this age
    gx_component_df = gx_component_df.loc[
        (gx_component_df['t_to_LISA'] <= gx_component_df['age']) &
        (gx_component_df['t_LISA_max'] >= gx_component_df['age'])
    ].copy()
    
    # first filter to make sure that we only keep DWDs that have formed by the present age
    gx_component_df = gx_component_df.loc[gx_component_df['age'] > gx_component_df['time']].copy()

    t_evol = gx_component_df['age'] - gx_component_df['time']
    
    a_today = lw.evol.evol_circ(
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun, 
        a_i=gx_component_df['semiMajor'].values * u.Rsun, 
        t_evol=t_evol.values * u.Myr,
        output_vars='a'
    ).to(u.Rsun).value[:,-1]
    gx_component_df['semiMajor_today'] = a_today

    # filter based on the semimajor axis that is within the LISA band
    a_LISA_hi = lw.utils.get_a_from_f_orb(
        f_orb=0.5*f_gw_LISA_high * u.Hz, 
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun
    ).to(u.Rsun).value

    a_LISA_lo = lw.utils.get_a_from_f_orb(
        f_orb=0.5*f_gw_LISA_low * u.Hz, 
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun
    ).to(u.Rsun).value

    # keep only DWDs that have a semimajor axis today that is within the LISA band
    gx_component_df = gx_component_df.loc[
        (gx_component_df['semiMajor_today'] < a_LISA_lo) & 
        (gx_component_df['semiMajor_today'] > a_LISA_hi)
    ]

    # keep only the DWDs that have not filled their Roche lobe at present day
    gx_component_df = gx_component_df.loc[
        gx_component_df['semiMajor_today'] > gx_component_df['a_rlo']
    ]

    return gx_component_df


def GetZ(RFin,iBin,MidRSet,ZCDFDictSet,n_draw):
    '''Returns the Z values for the DWDs in a galaxy component based on the Besancon parameters.
    
    Parameters
    ----------
    RFin : array-like
        Array of radial distances for the DWDs in the component.
    iBin : int
        Index of the component in the Besancon parameters.
    MidRSet : array-like
        Array of midpoints for the radial distribution of the component.
    ZCDFDictSet : dict
        Dictionary containing the vertical distribution parameters for the components.    
    n_draw : int
        Number of DWDs to draw Z values for.
    
    Returns
    -------
    zFin : array-like
        Array of Z values for the DWDs in the component.'''
    diffs = np.abs(MidRSet[None, :] - RFin[:, None])
    indices = np.argmin(diffs, axis=1)
    Xiz    = np.random.rand(n_draw)
    SignXi = np.sign(2*(np.random.rand(n_draw) - 0.5))
    zFin   = np.zeros(n_draw)
    # Loop over unique R-bin indices rather than individual DWDs
    for ind in np.unique(indices):
        mask   = indices == ind
        MidZSet = ZCDFDictSet[iBin+1][ind]['ZSet']
        RhozCDF = ZCDFDictSet[iBin+1][ind]['RhoCDFSet']
        zFin[mask] = SignXi[mask] * np.interp(Xiz[mask], RhozCDF, MidZSet)
    return zFin


def DrawRZ(iBin,n_draw,ModelRCache, ZCDFDictSet):
    '''Draws the R and Z values for the DWDs in a galaxy component based on the Besancon parameters.
    
    Parameters
    ----------
    iBin : int
        Index of the component in the Besancon parameters.
    n_draw : int
        Number of DWDs to draw R and Z values for.
    ModelRCache : dict
        Dictionary containing the radial distribution parameters for the galaxy components.
    ZCDFDictSet : dict
        Dictionary containing the vertical distribution parameters for the components.

    Returns
    -------
    RFin : array-like
        Array of R values for the DWDs in the component.
    zFin : array-like
        Array of Z values for the DWDs in the component.
    '''
    MidRSet    = ModelRCache[iBin]['MidRSet']
    RCDFSet    = ModelRCache[iBin]['RCDFSet']
    
    Xir        = np.random.rand(n_draw)
    RFin       = np.interp(Xir,RCDFSet,MidRSet)
    zFin       = GetZ(RFin,iBin,MidRSet,ZCDFDictSet,n_draw)
    
    ModelRCache = None
    ZCDFDictSet = None
    return RFin,zFin


def draw_positions(gx_component, n_samp, ModelRCache, ZCDFDictSet):
    '''Draws positions for the DWDs in a galaxy component based on the Besancon parameters.
    
    Parameters
    ----------
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    n_samp : int
        Number of DWDs in the Galaxy for this component.
    ModelRCache : dict
        Dictionary containing the radial distribution parameters for the galaxy components.
    ZCDFDictSet : dict
        Dictionary containing the vertical distribution parameters for the galaxy components.
    
    Returns
    -------
    positions : DataFrame
        DataFrame containing the positions of the DWDs in the component.
    '''
    iBin = utils.Besancon_params('BinName').tolist().index(gx_component)
    R, Z = DrawRZ(iBin, n_samp, ModelRCache, ZCDFDictSet)

    if gx_component == 'Bulge':

        alpha = utils.Besancon_params('Alpha')

        Th = np.random.uniform(0, 2*np.pi, n_samp)
        XPrime = R * np.cos(Th)
        YPrime = R * np.sin(Th)
        ZPrime = Z
        #ASSUMING THE ALPHA ANGLE IS ALONG THE GALACTIC ROTATION - CHECK DWEK
        X_set  = -ZPrime * np.sin(alpha) + XPrime * np.cos(alpha)
        Y_set  = ZPrime * np.cos(alpha) + XPrime * np.sin(alpha)
        Z_set  = -YPrime
        R_set  = np.sqrt(XPrime**2 + ZPrime**2)
    else:
        R_set = R
        Z_set = Z
        Th_set = np.random.uniform(0, 2*np.pi, n_samp)
        X_set = R_set * np.cos(Th_set)
        Y_set = R_set * np.sin(Th_set)
    
    X_rel     = X_set - utils.galaxy_params('RGalSun')
    Y_rel     = Y_set
    Z_rel     = Z_set + utils.galaxy_params('ZGalSun')
    
    dist     = np.sqrt(X_rel**2 + Y_rel**2 + Z_rel**2)
    Gal_b     = np.arcsin(Z_rel/dist)
    Gal_l     = np.zeros_like(Gal_b)

    disk_pos_mask = Y_rel>=0
    disk_neg_mask = Y_rel<0

    Gal_l[disk_pos_mask] = np.arccos(X_rel[disk_pos_mask]/(np.sqrt((dist[disk_pos_mask])**2 - (Z_rel[disk_pos_mask])**2)))
    Gal_l[disk_neg_mask]  = 2*np.pi - np.arccos(X_rel[disk_neg_mask]/(np.sqrt((dist[disk_neg_mask])**2 - (Z_rel[disk_neg_mask])**2)))
    
    positions = pd.DataFrame({
        'R': R_set,
        'Z': Z_set,
        'X_rel': X_rel,
        'Y_rel': Y_rel,
        'Z_rel': Z_rel,
        'X_gx': X_set,
        'Y_gx': Y_set,
        'Z_gx': Z_set,
        'dist': dist,
        'Gal_b': Gal_b,
        'Gal_l': Gal_l
    })
    
    return positions


def draw_metallicities(gx_component, n_samp):
    '''Draws metallicities for the DWDs in a galaxy component based on the Besancon parameters.
    
    Parameters
    ----------
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    n_samp : int
        Number of DWDs in the Galaxy for this component.
    
    Returns
    -------
    FeH : array-like
        Array of log(metallicities) assigned to the DWDs in the component.
    '''
    # For simplicity, we can assume a uniform distribution of metallicities
    # or use a predefined distribution based on the component.
    iBin = utils.Besancon_params('BinName').tolist().index(gx_component)
    mean_FeH = utils.Besancon_params('FeHMean')[iBin]
    std_FeH = utils.Besancon_params('FeHStD')[iBin]
    FeH = np.random.normal(mean_FeH,std_FeH, n_samp)
    
    return FeH


def create_galaxy_component(T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, f_LISA_low=1e-4, f_LISA_high=1e-1):
    '''Creates a DataFrame containing the DWDs in a specific galaxy component by sampling
    ages, metallicities, and positions based on the Besancon parameters.
    
    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    n_comp : int
        Number of DWDs in the Galaxy for this component.
    ModelRCache : dict
        Dictionary containing the radial distribution parameters for the galaxy components.
    ZCDFDictSet : dict
        Dictionary containing the vertical distribution parameters for the galaxy components.
    f_LISA_low : float, optional (default=1e-4)
        Lower bound of the LISA frequency band in Hz.
    f_LISA_high : float, optional (default=1e-1)
        Upper bound of the LISA frequency band in Hz.
    
    Returns
    -------
    gx_component_df : DataFrame
        DataFrame containing the DWDs in the specified galaxy component with assigned ages, metallicities, and positions.
    '''
    # Creates a DataFrame containing present-day DWDs in the Galaxy
    gx_component_df = T0_DWD_LISA.sample(n=n_comp, replace=True)
    
    # assign ages to the component based on the Besancon parameters
    gx_component_df['age'] = sample_component_ages(gx_component, n_samp=n_comp)

    # filter based on GW evolution up to the present day
    gx_component_df = filter_possible_LISA_sources(gx_component_df, f_LISA_low, f_LISA_high)
    
    # draw metallicities for the component
    gx_component_df['FeH'] = draw_metallicities(gx_component, n_samp=len(gx_component_df))

    # draw positions for the component
    positions = draw_positions(gx_component, n_samp=len(gx_component_df), ModelRCache=ModelRCache, ZCDFDictSet=ZCDFDictSet)
    gx_component_df = pd.concat([gx_component_df.reset_index(drop=True), positions.reset_index(drop=True)], axis=1)
    
    # clear memory
    del positions

    return gx_component_df


def create_galaxy_component_midpoint(T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, f_LISA_low=1e-4, f_LISA_high=1e-1, delta_t_bin=0.5):
    '''Creates a DataFrame containing the DWDs in a specific galaxy component by sampling
    ages, metallicities, and positions based on the Besancon parameters.
    
    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    n_comp : int
        Number of DWDs in the Galaxy for this component.
    ModelRCache : dict
        Dictionary containing the radial distribution parameters for the galaxy components.
    ZCDFDictSet : dict
        Dictionary containing the vertical distribution parameters for the galaxy components.
    f_LISA_low : float, optional (default=1e-4)
        Lower bound of the LISA frequency band in Hz.
    f_LISA_high : float, optional (default=1e-1)
        Upper bound of the LISA frequency band in Hz.
    delta_t_bin : float, optional (default=0.5 Myr)
        binwidth for age sampling; will assign ages at the midpoint of the bins
    
    Returns
    -------
    gx_component_df : DataFrame
        DataFrame containing the DWDs in the specified galaxy component with assigned ages, metallicities, and positions.
    '''
    # select the correct galaxy component index based on the Besancon parameters
    ii = utils.Besancon_params('BinName').tolist().index(gx_component)
    if ii < 0:
        raise ValueError(f"Component {gx_component} not found in Besancon parameters.")
    
    # Get the age range and mass fraction for the component
    t_lo = utils.Besancon_params('AgeMin')[ii]
    t_hi = utils.Besancon_params('AgeMax')[ii]

    if t_lo == t_hi:
        # filter the T0_DWD_LISA DataFrame to only include DWDs that have a time to LISA 
        # band that falls within the age bins of the component
        T0_DWD_LISA_filter = T0_DWD_LISA.loc[
            (T0_DWD_LISA['t_to_LISA'] < t_hi) &
            (T0_DWD_LISA['t_LISA_max'] > t_hi)].copy()
        
        # draw all n_comp samples from the filtered pool active at this age
        n_samp = len(T0_DWD_LISA_filter) / len(T0_DWD_LISA) * n_comp
        # probabilistically add one more sample if the fractional part 
        # is greater than a random number
        n_samp = int(n_samp) + (np.random.uniform() < (n_samp % 1))

        if n_samp > 0 and len(T0_DWD_LISA_filter) > 0:
            # Creates a DataFrame containing present-day DWDs in the Galaxy
            gx_component_df = T0_DWD_LISA_filter.sample(n=int(n_samp), replace=True)
            # assign ages to the component based on the Besancon parameters
            gx_component_df['age'] = np.ones(n_samp) * t_lo
        
            # filter based on GW evolution up to the present day
            gx_component_df = filter_possible_LISA_sources(gx_component_df, f_LISA_low, f_LISA_high)
        
            # draw metallicities for the component
            gx_component_df['FeH'] = draw_metallicities(gx_component, n_samp=len(gx_component_df))
        
            # draw positions for the component
            positions = draw_positions(gx_component, n_samp=len(gx_component_df), ModelRCache=ModelRCache, ZCDFDictSet=ZCDFDictSet)
            gx_component_df = pd.concat([gx_component_df.reset_index(drop=True), positions.reset_index(drop=True)], axis=1)
            
            # clear memory
            del positions

            # log the numbers
            n_samp_list = [n_samp]
            n_filter_list = [len(gx_component_df)]
            age_list = [t_hi]
    else:
        # set up bins between minimum age and maximum age of the galaxy component

        age_bins = np.arange(t_lo, t_hi + delta_t_bin/2, delta_t_bin)
        midpoints = age_bins[:-1] + delta_t_bin/2  # shape: (n_bins,)
        
        t_to_LISA = T0_DWD_LISA['t_to_LISA'].values  # shape: (n_DWD,)
        t_LISA_max = T0_DWD_LISA['t_LISA_max'].values  # shape: (n_DWD,)
        
        # Broadcasting: compare every DWD against every midpoint at once
        # active_mask shape: (n_DWD, n_bins)
        active_mask = (
            (t_to_LISA[:, None] < midpoints[None, :]) &
            (t_LISA_max[:, None] > midpoints[None, :])
        )
        
        # Get flat indices of active (DWD, midpoint) pairs
        dwd_idx, mid_idx = np.where(active_mask)  # both shape: (n_active_pairs,)
    
        # Build the expanded DataFrame — one row per active (DWD, midpoint) pair
        expanded = T0_DWD_LISA.iloc[dwd_idx].copy().reset_index(drop=True)
        expanded['age'] = midpoints[mid_idx]
        expanded['_mid_idx'] = mid_idx  # optional: keep bin label for sampling weights

        # total expected samples across all bins (before stochastic rounding)
        n_active_DWDs = len(expanded)           # all (DWD, midpoint) DWDs that are active at some midpoint
        n_total = len(T0_DWD_LISA)
        n_bins = len(midpoints)
        
        # each pair gets equal weight; n_comp drawn proportional to active fraction
        expected_total = n_comp * n_active_DWDs / (n_bins * n_total)
        n_draw = int(expected_total) + (np.random.uniform() < (expected_total % 1))
        
        # single sample over the entire expanded pool
        gx_component_df = expanded.sample(n=n_draw, replace=True)
        n_samp_per_bin = np.bincount(gx_component_df['_mid_idx'].values, minlength=len(midpoints))

        # filter based on GW evolution up to the present day
        gx_component_df = filter_possible_LISA_sources(gx_component_df, f_LISA_low, f_LISA_high)
        n_filter_per_bin = np.bincount(gx_component_df['_mid_idx'].values, minlength=len(midpoints))

        # draw metallicities for the component
        gx_component_df['FeH'] = draw_metallicities(gx_component, n_samp=len(gx_component_df))

        # draw positions for the component
        positions = draw_positions(gx_component, n_samp=len(gx_component_df), ModelRCache=ModelRCache, ZCDFDictSet=ZCDFDictSet)
        gx_component_df = pd.concat([gx_component_df.reset_index(drop=True), positions.reset_index(drop=True)], axis=1)
        del positions

        # log the numbers
        # only keep bins that had any samples
        active_bins = n_samp_per_bin > 0
        n_samp_list = n_samp_per_bin[active_bins].tolist()
        n_filter_list = n_filter_per_bin[active_bins].tolist()
        age_list = midpoints[active_bins].tolist()
    
    return gx_component_df, (n_samp_list, n_filter_list, age_list)

def get_legwork_calculations(gx, t_obs=4 * u.yr):
    '''Calculates the legwork SNR and h0 for the DWDs in a galaxy component.

    Parameters
    ----------
    gx : DataFrame
        DataFrame containing the DWDs in a galaxy component.
    t_obs : Quantity, optional
        Observation time for the legwork calculations. Default is 4 years.

    Returns
    -------
    gx : DataFrame
        DataFrame with legwork SNR and h0 calculated for each DWD.
    '''
    # Calculate the h0 strain for each DWD
    sources = lw.source.Source(
        m_1=gx['mass1'].values * u.Msun, 
        m_2=gx['mass2'].values * u.Msun, 
        ecc=np.zeros(len(gx)), 
        dist=gx['dist'].values * u.kpc, 
        f_orb=lw.utils.get_f_orb_from_a(
            a=gx['semiMajor_today'].values * u.Rsun,
            m_1=gx['mass1'].values * u.Msun,
            m_2=gx['mass2'].values * u.Msun
            ).to(u.Hz),
        interpolate_g=False
        )
    
    gx['h0'] = sources.get_h_c_n(harmonics=[2]).flatten()
    gx['legwork_SNR'] = sources.get_snr(t_obs=t_obs, confusion_noise=None)
    
    return gx

def write_galaxy(gx, write_path, gx_component, stats, cols_DWDs, write_h5=False, verbose=False):
    '''Writes the galaxy DataFrame to a file.
    
    Parameters
    ----------
    gx : DataFrame
        DataFrame containing the DWDs in the Galaxy.
    write_path : str
        Path to save the DataFrame containing the DWDs in the Galaxy.
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    stats : DataFrame
        DataFrame containing the number of LISA DWDs for the component.
    cols_DWDs : list
        List of column names for the DWDs in the galaxy DataFrame.
    write_h5 : bool, optional
        If True, writes the DataFrame to an HDF5 file. If False, writes to a CSV file. Default is False.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.
    
    Returns
    -------
    None
    '''
    if write_path is None:
        raise RuntimeError("No write path specified. Galaxy will not be saved.")

    legwork_mask = gx['legwork_SNR'] > 7

    if write_h5:
        if verbose:
            print(f"Writing {len(gx)} DWDs for component {gx_component} to {write_path}_Galaxy_AllDWDs.h5")
        gx[cols_DWDs].to_hdf(write_path+'_Galaxy_AllDWDs.h5', mode='a', append=True, key='AllDWDs', format='table')

        if verbose:
            print(f"Writing {len(gx[legwork_mask])} LISA DWDs for component {gx_component} to {write_path}_Galaxy_LISA_DWDs.h5")
        gx[legwork_mask][cols_DWDs].to_hdf(write_path+'_Galaxy_LISA_DWDs.h5', mode='a', append=True, key='LISA_DWDs', format='table')

        if verbose:
            print(f"Writing statistics for component {gx_component} to {write_path}_Galaxy_LISA_Candidates_Bin_Data.h5")
        stats.to_hdf(write_path+'_Galaxy_LISA_Candidates_Bin_Data.h5', mode='a', append=True, key='BinData', format='table')

    else:
        if verbose:
            print(f"Writing {len(gx)} DWDs for component {gx_component} to {write_path}_Galaxy_AllDWDs.csv")
        if not os.path.exists(write_path+'_Galaxy_AllDWDs.csv'):
            gx[cols_DWDs].to_csv(write_path+'_Galaxy_AllDWDs.csv', mode='w', index=False)
        else:
            gx[cols_DWDs].to_csv(write_path+'_Galaxy_AllDWDs.csv', mode='a', header=False, index=False)

        if verbose:
            print(f"Writing {len(gx[legwork_mask])} LISA DWDs for component {gx_component} to {write_path}_Galaxy_LISA_DWDs.csv")
        if not os.path.exists(write_path+'_Galaxy_LISA_DWDs.csv'):
            gx[legwork_mask][cols_DWDs].to_csv(write_path+'_Galaxy_LISA_DWDs.csv', mode='w', index=False)
        else:
            gx[legwork_mask][cols_DWDs].to_csv(write_path+'_Galaxy_LISA_DWDs.csv', mode='a', header=False, index=False)

        if verbose:
            print(f"Writing statistics for component {gx_component} to {write_path}_Galaxy_LISA_Candidates_Bin_Data.csv")
        if not os.path.exists(write_path+'_Galaxy_LISA_Candidates_Bin_Data.csv'):
            stats.to_csv(write_path+'_Galaxy_LISA_Candidates_Bin_Data.csv', mode='w', index=False)
        else:
            stats.to_csv(write_path+'_Galaxy_LISA_Candidates_Bin_Data.csv', mode='a', header=False, index=False)

    del gx
    return None

def create_LISA_galaxy(T0_DWD_LISA, N_DWD_Gx, ModelParams, write_path):
    '''Creates a DataFrame containing present-day DWDs in the Galaxy
    that have frequencies in the LISA band 
    
    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    N_DWD_Gx : int
        Number of DWDs in the Galaxy for this component.
    ModelParams : dict
        Dictionary containing model parameters including 'run_sub_type'.
    write_path : str
        Path to save the DataFrame containing the DWDs in the Galaxy.

    Returns
    -------
    gx_tot : DataFrame
        DataFrame containing the DWDs in the specified component with assigned ages.
    '''
    if ModelParams['verbose']:
        import tqdm

    # Build the file prefix from the directory path and code name
    write_path = os.path.join(write_path, ModelParams['code'])
    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    # Load the radial and vertical distribution parameters for the galaxy components
    ModelRCache = utils.load_Rdicts_from_hdf5('./GalCache/BesanconRData.h5')
    ZCDFDictSet = utils.load_RZdicts_from_hdf5('./GalCache/BesanconRZData.h5')

    if ModelParams['verbose']:
        print(f"Creating a Galaxy with {N_DWD_Gx} DWDs total. For verbose output, set verbose=True in create_galaxy function call.")
    if ModelParams['verbose']:
        iterator = tqdm.tqdm(utils.Besancon_params('BinName'))
    else:
        iterator = utils.Besancon_params('BinName')
    loop_length = ModelParams.get('loop_length', 1e6)
    for ii, gx_component in enumerate(iterator):
        n_comp = N_DWD_Gx * utils.Besancon_params('BinMassFractions')[ii]
        
        # Can only sample an integer number of DWDs, so we take the integer part and add one
        # if a random number is less than the fractional part
        n_comp = int(n_comp) + (np.random.uniform() < (n_comp % 1))
        if ModelParams['verbose']:
            print(f"Sampling {n_comp} DWDs for component {gx_component} ({ii+1}/{len(utils.Besancon_params('BinName'))})")
        if n_comp <= 0:
            print(f"Component {gx_component} has no DWDs to sample")
        
        if n_comp > loop_length:
            # If the number of DWDs to sample is too large, we will loop over the sampling
            # to avoid memory issues.
            n_loop = int(n_comp / loop_length)
            n_left_over = int(n_comp - n_loop * loop_length)
            
            gx_component_df = pd.DataFrame()
            n_comp = loop_length
            if ModelParams['verbose']:
                print(f"Reducing number of DWDs to sample for component {gx_component} by looping {n_loop} times with {n_comp}")

            # do the looping
            stats_tot = pd.DataFrame()    
            for ii in range(n_loop):
                # create the galaxy component DataFrame
                if ModelParams['midpoint_ages']:
                    gx, stats_list = create_galaxy_component_midpoint(
                        T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'], ModelParams['delta_t_gal_myr'])
                else:
                    gx = create_galaxy_component(
                        T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'])

                # get the statistical data of number of DWDs in a finer time sampling grid
                stats = get_component_stats(gx_component, gx, ModelParams)
                if stats_tot.empty:
                    stats_tot = stats
                else:
                    stats_tot['n_DWDs'] = stats_tot['n_DWDs'].values + stats['n_DWDs'].values
                
                # Calculate the strain amplitude and SNR without confusion for each DWD in the component
                gx = get_legwork_calculations(gx)

                # assign the component name to the DataFrame
                gx['component'] = gx_component

                # write the sub-loop
                _ = write_galaxy(gx, write_path, gx_component, stats, cols_DWDs=ModelParams['cols_write'], write_h5=ModelParams['write_h5'], verbose=ModelParams['verbose'])  

                # append the stats list for this loop to the overall stats list for the component
                if ModelParams['midpoint_ages']:
                    if ii == 0:
                        stats_list_tot = stats_list
                    else:
                        stats_list_tot = [x + y for x, y in zip(stats_list_tot, stats_list)]
                
            # create the last galaxy component DataFrame with the left over DWDs
            if ModelParams['verbose']:
                print(f"Adding {n_left_over} left over DWDs for component {gx_component}")
            
            if ModelParams['midpoint_ages']:
                gx, stats_list = create_galaxy_component_midpoint(
                    T0_DWD_LISA, gx_component, n_left_over, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'], ModelParams['delta_t_gal_myr'])
                stats_list_tot = [x + y for x, y in zip(stats_list_tot, stats_list)]

            else:
                gx = create_galaxy_component(T0_DWD_LISA, gx_component, n_left_over, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'])

            # get the statistical data of number of DWDs in a finer time sampling grid
            stats = get_component_stats(gx_component, gx, ModelParams)
            stats_tot['n_DWDs'] = stats_tot['n_DWDs'].values + stats['n_DWDs'].values

            # Calculate the strain amplitude and SNR without confusion for each DWD in the component
            gx = get_legwork_calculations(gx)

            # assign the component name to the DataFrame
            gx['component'] = gx_component

            # write the sub-loop
            _ = write_galaxy(gx, write_path, gx_component, stats, cols_DWDs=ModelParams['cols_write'], write_h5=ModelParams['write_h5'], verbose=ModelParams['verbose'])

            # append the stats list to a data file for the component
            if ModelParams['midpoint_ages']:
                stats_list_df = pd.DataFrame({
                    'age': stats_list_tot[2],
                    'n_samp': stats_list_tot[0],
                    'n_filter': stats_list_tot[1],
                    'component': gx_component
                })
                if ModelParams['write_h5']:
                    stats_list_df.to_hdf(write_path + '_midpoint_stats.h5', mode='a', append=True, key='midpoint_stats', format='table')
                else:
                    if not os.path.exists(write_path + '_midpoint_stats.csv'):
                        stats_list_df.to_csv(write_path + '_midpoint_stats.csv', mode='w', index=False)
                    else:
                        stats_list_df.to_csv(write_path + '_midpoint_stats.csv', mode='a', header=False, index=False)

        else:
            # create the galaxy component DataFrame
            if ModelParams['midpoint_ages']:
                gx_component_df, stats_list = create_galaxy_component_midpoint(
                    T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'], ModelParams['delta_t_gal_myr'])
            else:
                gx_component_df = create_galaxy_component(
                    T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet, ModelParams['f_LISA_low'], ModelParams['f_LISA_high'])

            # get the statistical data of number of DWDs in a finer time sampling grid
            stats_tot = get_component_stats(gx_component, gx_component_df, ModelParams)

            # Calculate the strain amplitude and SNR without confusion for each DWD in the component
            gx_component_df = get_legwork_calculations(gx_component_df)
            
            # assign the component name to the DataFrame
            gx_component_df['component'] = gx_component
            
            # write the galaxy component DataFrame to a file
            _ = write_galaxy(gx_component_df, write_path, gx_component, stats_tot, cols_DWDs=ModelParams['cols_write'], write_h5=ModelParams['write_h5'], verbose=ModelParams['verbose'])

            # write the midpoint stats list to a file
            if ModelParams['midpoint_ages']:
                stats_list_df = pd.DataFrame({
                    'age': stats_list[2],
                    'n_samp': stats_list[0],
                    'n_filter': stats_list[1],
                    'component': gx_component
                })
                if ModelParams['write_h5']:
                    stats_list_df.to_hdf(write_path + '_midpoint_stats.h5', mode='a', append=True, key='midpoint_stats', format='table')
                else:
                    if not os.path.exists(write_path + '_midpoint_stats.csv'):
                        stats_list_df.to_csv(write_path + '_midpoint_stats.csv', mode='w', index=False)
                    else:
                        stats_list_df.to_csv(write_path + '_midpoint_stats.csv', mode='a', header=False, index=False)
            

    return None