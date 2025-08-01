import astropy.units as u
import numpy as np
import legwork as lw
import pandas as pd
from rapid_code_load_T0 import load_T0_data
import utils

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

    CodeDir = os.path.dirname(os.path.abspath(__file__))

    # Load the mass-radius data for white dwarfs
    M_WD_dat, R_WD_dat = np.split(np.loadtxt(CodeDir + '/WDData/MRRel.dat'),2,axis=1)
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
        Dictionary containing model parameters including 'RunSubType'.
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
    T0_data, _ = load_T0_data(T0_dat_path)
    
    if verbose:
        print(f"Loaded T0 data with {len(T0_data)} entries.")

    # Handle the DWD selection based on the Code in ModelParams
    # and apply a semimajor axis cut as specified in ModelParams
    if ModelParams['Code'] == 'ComBinE':
        T0_DWD =  T0_data.loc[
            (T0_data.type1 == 2) & (T0_data.type2 == 2) & 
            (T0_data.semiMajor > 0) & (T0_data.semiMajor < ModelParams['ACutRSunPre'])
            ].groupby('ID', as_index=False).first()
    else:
        T0_DWD = T0_data.loc[
            (T0_data.type1.isin([21,22,23])) & (T0_data.type2.isin([21,22,23])) & 
            (T0_data.semiMajor > 0) & (T0_data.semiMajor < ModelParams['ACutRSunPre'])
            ].groupby('ID', as_index=False).first() 
        
    if verbose: 
        print(f"Found {len(T0_DWD)} DWDs in T0 data with semimajor axis < {ModelParams['ACutRSunPre']} RSun.")
            
    return T0_DWD


def get_a_RLO(T0_DWD):
    '''Calculates the semimajor axis at Roche overflow for each DWD in the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.

    Returns
    -------
    T0_DWD : DataFrame
        DataFrame with Roche lobe radius added for each DWD.
    '''
    
    # First determine the separation where the donor fills its Roche lobe
    # Calculate the radius of the least massive WD 
    # [this will be the donor at mass transfer]
    T0_DWD['R_don'] = get_R_WD(np.minimum(T0_DWD['mass1'].values,T0_DWD['mass2'].values))

    # Calculate the mass ratio where q = don/acc
    T0_DWD['q_rlo'] = np.minimum(T0_DWD['mass1'], T0_DWD['mass2']) / np.maximum(T0_DWD['mass1'], T0_DWD['mass2'])

    # Calculate the Roche lobe radius in RSun
    T0_DWD['a_rlo'] = frac_RL(T0_DWD['q_rlo'].values) * T0_DWD['semiMajor'].values

    return T0_DWD


def get_a_LISA(T0_DWD, f_LISA_low=1e-4): 
    '''Calculates the semimajor axis at the lower bound of LISA 
    frequency for each DWD in the T0 data.
    Parameters
    ---------- 
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.
        
    Returns
    -------
    T0_DWD : DataFrame
        DataFrame with LISA semimajor axis added for each DWD.
    '''
    # Calculate the semimajor axis at the lower bound of LISA sensitivity frequency
    T0_DWD['a_LISA'] = lw.utils.get_a_from_f_orb(
        f_orb=f_LISA_low * u.Hz, 
        m_1=T0_DWD.mass1.values * u.Msun, 
        m_2=T0_DWD.mass2.values * u.Msun
    ).to(u.Rsun).value
    
    return T0_DWD


def get_GW_timescales(T0_DWD):
    '''Calculates the GW inspiral timescales for each DWD in the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.

    Returns
    -------
    T0_DWD : DataFrame
        DataFrame with GW inspiral timescales added for each DWD.
    '''
    
    # Calculate the time to merger from DWD formation
    T0_DWD['t_merge_gw'] = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['semiMajor'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calculate the time to merger from the Roche lobe overflow
    T0_DWD['t_merge_rlo'] = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['a_rlo'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calculate the time to merger from the LISA semimajor axis
    T0_DWD['t_merge_lisa'] = lw.evol.get_t_merge_circ(
        m_1=T0_DWD['mass1'].values * u.Msun, 
        m_2=T0_DWD['mass2'].values * u.Msun, 
        a_i=T0_DWD['a_LISA'].values * u.Rsun
        ).to(u.Myr).value
    
    # Calcalate the time to get to the LISA band from formation
    T0_DWD['t_to_LISA'] = T0_DWD['time'] + (T0_DWD['t_merge_gw'] - T0_DWD['t_merge_lisa']).clip(lower=0)

    # Calculate the time to get to Roche lobe overflow from bottom of LISA band
    T0_DWD['t_LISA_max'] = T0_DWD['time'] + (T0_DWD['t_merge_gw'] - T0_DWD['t_merge_rlo']).clip(lower=0)
    
    return T0_DWD


def filter_for_potential_LISA_sources(T0_DWD, ModelParams, verbose=False):
    '''Calculates the properties of the DWDs based on the T0 data.
    
    Parameters
    ----------
    T0_DWD : DataFrame
        DataFrame containing the T0 data for DWDs.
    ModelParams : dict
        Dictionary containing model parameters including 'MaxTDelay'.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.

    Returns
    -------
    T0_DWD : DataFrame
        DataFrame with calculated properties for each DWD.
    '''
    
    # get the semimajor axis at Roche lobe overflow
    T0_DWD = get_a_RLO(T0_DWD)

    # get the semimajor axis at the lower bound of LISA frequency
    T0_DWD = get_a_LISA(T0_DWD)

    # get the GW inspiral timescales
    T0_DWD = get_GW_timescales(T0_DWD)

    # filter based on sources that don't make it to LISA band before max age
    T0_DWD = T0_DWD.loc[T0_DWD['t_to_LISA'] < ModelParams['MaxTDelay']]

    if verbose:
        print(f"Filtered DWDs to {len(T0_DWD)} that will evolve to LISA band before {ModelParams['MaxTDelay']} Myr.")

    return T0_DWD


def get_possible_T0_LISA_sources(ModelParams, T0_dat_path, verbose=False):
    '''Returns a list of likely LISA sources drawn from the 
    T0 data and the maximum Galactic component age 
    specified in ModelParams.
    
    Parameters
    ----------
    ModelParams : dict
        Dictionary containing model parameters including 'RunSubType' and 'MaxTDelay'.
    T0_dat_path : str
        Path to the T0 data file.
    verbose : bool, optional
        If True, prints additional information during processing. Default is False.
    
    Returns
    -------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    '''

    # I think we want to move this outside
    # Based on the model parameters, determine the mass that was initialized for the T0 data
    # and the effective number of DWDs in the Galaxy per simulated DWD based on the mass
    #MassNorm        = get_mass_norm(ModelParams['RunSubType'])
    #NStarsPerRun    = GalaxyParams['MGal']/MassNorm

    # Get the T0 data for DWDs
    # Note that the path should be the path to 
    # 'data_products' directory in the Google Drive from rclone
    T0_DWD = get_T0_DWDs(ModelParams, T0_dat_path, verbose=verbose)
    if T0_DWD.empty:
        raise ValueError("No DWDs found in the T0 data. Check the input parameters or data file.")

    # Calculate orbital properties and GW evolution timescales based on the T0 WDs
    # This will also filter out DWDs that will not evolve to the LISA band before 
    # the maximum age specified in ModelParams['MaxTDelay']
    T0_DWD_LISA = filter_for_potential_LISA_sources(T0_DWD, ModelParams, verbose=verbose)
    if T0_DWD_LISA.empty:
        raise ValueError("No DWDs found that will evolve to the LISA band before the maximum age. Check the input parameters or T0 data file.")

    return T0_DWD_LISA

def get_N_Gx_sample(T0_DWD_LISA, ModelParams):
    '''Returns the number of DWDs in the Galaxy based on the model parameters.

    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    ModelParams : dict
        Dictionary containing model parameters including 'RunSubType'.
    
    Returns
    -------
    N_DWD_Gx : int
        Number of DWDs in the Galaxy.
    '''
    mass_norm  = utils.get_mass_norm(IC_model=ModelParams['RunSubType'], binary_fraction=0.5)
    gx_to_sim_mass = utils.galaxy_params('MGal')/mass_norm
    
    N_DWD_Gx = len(T0_DWD_LISA) * gx_to_sim_mass
    N_DWD_Gx = int(N_DWD_Gx) + (np.random.uniform() < (N_DWD_Gx % 1))
    
    return N_DWD_Gx


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
    ages = np.random.uniform(t_lo, t_hi, n_samp)
    return ages

def filter_LISA_sources(gx_component_df):
    '''Filters the DWDs in a galaxy component DataFrame to those that are likely LISA sources.
    
    Parameters
    ----------
    gx_component_df : DataFrame
        DataFrame containing the DWDs in a specific galaxy component.
    
    Returns
    -------
    gx_component_df : DataFrame
        DataFrame containing only the DWDs that have a semimajor axis within the LISA band
    '''
    # calculate the evolution of the orbit due to GW emission
    gx_component_df = gx_component_df.loc[gx_component_df['age'] > gx_component_df['time']].copy()
    t_evol = gx_component_df['age'] - gx_component_df['time']
    
    a_today = lw.evol.evol_circ(
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun, 
        a_i=gx_component_df['semiMajor'].values * u.Rsun, 
        t_evol=t_evol.values * u.Myr,
        output_vars='a'
    ).to(u.Rsun).value

    gx_component_df['semiMajor_today'] = a_today[:,-1]

    # filter based on the semimajor axis that is within the LISA band
    a_LISA_hi = lw.utils.get_a_from_f_orb(
        f_orb=1e-1 * u.Hz, 
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun
    ).to(u.Rsun).value

    a_LISA_lo = lw.utils.get_a_from_f_orb(
        f_orb=1e-4 * u.Hz, 
        m_1=gx_component_df['mass1'].values * u.Msun, 
        m_2=gx_component_df['mass2'].values * u.Msun
    ).to(u.Rsun).value

    gx_component_df = gx_component_df.loc[
        (gx_component_df['semiMajor_today'] < a_LISA_lo) & 
        (gx_component_df['semiMajor_today'] > a_LISA_hi)
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
    indices = np.argmin(diffs, axis=1).tolist()
    zFin = np.zeros(n_draw)
    # Loop through the indices to get the Z values
    for ii, ind in enumerate(indices):
        # CDFs start at 1
        MidZSet = ZCDFDictSet[iBin+1][ind]['ZSet']
        RhozCDF = ZCDFDictSet[iBin+1][ind]['RhoCDFSet']
    
        Xiz        = np.random.rand()
        SignXi     = np.sign(2*(np.random.rand() - 0.5))
        zFin[ii]   = SignXi*np.interp(Xiz,RhozCDF,MidZSet)   
    return np.array(zFin)

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
        'R': R,
        'Z': Z,
        'X_rel': X_rel,
        'Y_rel': Y_rel,
        'Z_rel': Z_rel,
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
    metallicities : array-like
        Array of metallicities assigned to the DWDs in the component.
    '''
    # For simplicity, we can assume a uniform distribution of metallicities
    # or use a predefined distribution based on the component.
    iBin = utils.Besancon_params('BinName').tolist().index(gx_component)
    mean_FeH = utils.Besancon_params('FeHMean')[iBin]
    std_FeH = utils.Besancon_params('FeHStD')[iBin]
    FeH = np.random.normal(mean_FeH,std_FeH, n_samp)
    return FeH


def create_galaxy_component(T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet):
    # Creates a DataFrame containing present-day DWDs in the Galaxy
    gx_component_df = T0_DWD_LISA.sample(n=n_comp, replace=True)
    # assign ages to the component based on the Besancon parameters
    gx_component_df['age'] = sample_component_ages(gx_component, n_samp=n_comp)
    
    # filter based on GW evolution up to the present day
    gx_component_df = filter_LISA_sources(gx_component_df)

    # draw metallicities for the component
    gx_component_df['FeH'] = draw_metallicities(gx_component, n_samp=len(gx_component_df))

    # draw positions for the component
    positions = draw_positions(gx_component, n_samp=len(gx_component_df), ModelRCache=ModelRCache, ZCDFDictSet=ZCDFDictSet)
    gx_component_df = pd.concat([gx_component_df.reset_index(drop=True), positions.reset_index(drop=True)], axis=1)
    positions = None # clear memory

    return gx_component_df
    
def create_LISA_galaxy(T0_DWD_LISA, N_DWD_Gx, write_path, verbose=True):
    '''Creates a DataFrame containing present-day DWDs in the Galaxy
    that have frequencies in the LISA band 
    
    Parameters
    ----------
    T0_DWD_LISA : DataFrame
        DataFrame containing the T0 data for DWDs that are likely LISA sources.
    gx_component : str
        Name of the component (e.g., 'ThinDisk1', 'ThinDisk2', etc.).
    N_DWD_Gx : int
        Number of DWDs in the Galaxy for this component.
    write_path : str
        Path to save the DataFrame containing the DWDs in the Galaxy.
    verbose : bool, optional
        If True, prints additional information during processing. Default is True.    
    Returns
    -------
    gx_tot : DataFrame
        DataFrame containing the DWDs in the specified component with assigned ages.
    '''
    import tqdm

    # Load the radial and vertical distribution parameters for the galaxy components
    ModelRCache = utils.load_Rdicts_from_hdf5('./GalCache/BesanconRData.h5')
    ZCDFDictSet = utils.load_RZdicts_from_hdf5('./GalCache/BesanconRZData.h5')


    for ii, gx_component in enumerate(tqdm.tqdm(utils.Besancon_params('BinName'))):
        n_comp = N_DWD_Gx * utils.Besancon_params('BinMassFractions')[ii]

        # Can only sample an integer number of DWDs, so we take the integer part and add one
        # if a random number is less than the fractional part
        n_comp = int(n_comp) + (np.random.uniform() < (n_comp % 1))
        if verbose:
            print(f"Sampling {n_comp} DWDs for component {gx_component} ({ii+1}/{len(utils.Besancon_params('BinName'))})")
        if n_comp <= 0:
            print(f"Component {gx_component} has no DWDs to sample")
        
        
        loop_length = 1e6
        if n_comp > loop_length:
            # If the number of DWDs to sample is too large, we will loop over the sampling
            # to avoid memory issues.
            n_loop = int(n_comp / loop_length)
            n_left_over = int(n_comp - n_loop * loop_length)
            
            gx_component_df = pd.DataFrame()
            n_comp = int(n_comp / n_loop)
            if verbose:
                print(f"Reducing number of DWDs to sample for component {gx_component} by looping {n_loop} times with {n_comp}")

            # do the looping    
            for ii in range(n_loop):
                # create the galaxy component DataFrame
                gx = create_galaxy_component(T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet)
                if gx_component_df.empty:
                    gx_component_df = gx
                else:
                    gx_component_df = pd.concat([gx_component_df, gx], ignore_index=True)
                gx = None # clear memory
            
            # create the last galaxy component DataFrame with the left over DWDs
            if verbose:
                print(f"Adding {n_left_over} left over DWDs for component {gx_component}")
            gx = create_galaxy_component(T0_DWD_LISA, gx_component, n_left_over, ModelRCache, ZCDFDictSet)
            gx_component_df = pd.concat([gx_component_df, gx], ignore_index=True)
        else:
            # create the galaxy component DataFrame
            gx_component_df = create_galaxy_component(T0_DWD_LISA, gx_component, n_comp, ModelRCache, ZCDFDictSet)
        
        # add the component name to the DataFrame
        gx_component_df['component'] = gx_component

        # Calculate the strain amplitude for each DWD in the component
        gx_component_df['h0'] = lw.strain.h_0_n(
            m_c=lw.utils.chirp_mass(m_1=gx_component_df['mass1'].values * u.Msun,
                                    m_2=gx_component_df['mass2'].values * u.Msun),
            f_orb=lw.utils.get_f_orb_from_a(
                a=gx_component_df['semiMajor_today'].values * u.Rsun,
                m_1=gx_component_df['mass1'].values * u.Msun,
                m_2=gx_component_df['mass2'].values * u.Msun
            ).to(u.Hz),
            ecc=np.zeros(len(gx_component_df)),
            dist=gx_component_df['dist'].values * u.kpc,
            n=2
        ).flatten()
        
        
        if write_path is not None:
            if verbose:
                print(f"Writing {len(gx_component_df)} DWDs for component {gx_component} to {write_path}")
            # Save the galaxy component DataFrame to the specified path
            gx_component_df.to_hdf(write_path, mode='a', append=True, key='LISA_DWDs', format='table')
        else:
            raise Warning("No write path specified. Galaxy will not be saved.")
        # clear the gx_component_df to save memory
        gx_component_df = None

    return None