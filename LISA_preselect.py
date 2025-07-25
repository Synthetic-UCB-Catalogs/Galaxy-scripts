import astropy.units as u
import numpy as np
import legwork as lw
from rapid_code_load_T0 import load_T0_data
from utils import get_mass_norm


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


def calc_filter_properties(T0_DWD, ModelParams, verbose=False):
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
    T0_DWD_LISA = calc_filter_properties(T0_DWD, ModelParams, verbose=verbose)
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
    mass_norm  = get_mass_norm(IC_model, binary_fraction=0.5)
    NStarsPerRun    = GalaxyParams['MGal']/MassNorm
    
    
    return N_DWD_Gx