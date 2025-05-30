import numpy as np
import pandas as pd
import os

def dwd_count_single_code(code_name, icv_name, rclone_flag=True):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code/variation. If rclone_flag is True, filepaths assume you have set up
    rclone for the project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    icv_name: str
        Name of the initial condition variation (e.g. "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    dwd_count: float
        Number of LISA DWDs predicted in the Galaxy for that code/variation.
    """
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/initial_condition_variations/'
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/initial_condition_variations/'
    all_dwd_filepath = initial_string + icv_name + '/' + code_name + \
        '_Galaxy_AllDWDs.csv'
    bin_data_filepath = initial_string + icv_name + '/' + code_name + \
        '_Galaxy_LISA_Candidates_Bin_Data.csv'
    lisa_dwd_filepath = initial_string + icv_name + '/' + code_name + \
        '_Galaxy_LISA_DWDs.csv'
    
    all_dwd_array = pd.read_csv(all_dwd_filepath)
    bin_data_array = pd.read_csv(bin_data_filepath)
    lisa_dwd_array = pd.read_csv(lisa_dwd_filepath)
    
    real_dwd_multiplier = sum(bin_data_array['SubBinNDWDsReal'])
    dwd_count = real_dwd_multiplier * len(lisa_dwd_array) / len(all_dwd_array)
    
    return dwd_count

def dwd_count_icv_average(code_name, rclone_flag=True):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code, averaged over each initial condition variation.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    mean_dwd_count: float
        Number of LISA DWDs predicted in the Galaxy for that code, averaged
        over all initial condition variations.
    """
    
    icv_names = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01', \
                 'thermal_ecc', 'uniform_ecc']
    var_count = np.empty((len(icv_names))) #holds counts from each IC variation
    
    for i in range(len(icv_names)):
        var_count[i] = dwd_count_single_code(code_name, icv_names[i], \
                                             rclone_flag)
    
    mean_dwd_count = np.mean(var_count) #average counts over IC variations
    
    return mean_dwd_count

def dwd_count_icv_min_max(code_name, rclone_flag=True):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code, and returns the minimum and maximum values across the different
    initial condition variations.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    min_dwd_count: float
        Minimum number of LISA DWDs predicted in the Galaxy for that code over
        all initial condition variations.
    max_dwd_count: float
        Minimum number of LISA DWDs predicted in the Galaxy for that code over
        all initial condition variations.
    """
    
    icv_names = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01', \
                 'thermal_ecc', 'uniform_ecc']
    var_count = np.empty((len(icv_names))) #holds counts from each IC variation
    
    for i in range(len(icv_names)):
        var_count[i] = dwd_count_single_code(code_name, icv_names[i], \
                                             rclone_flag)
    
    min_dwd_count = np.min(var_count)
    max_dwd_count = np.min(var_count)
    
    return min_dwd_count, max_dwd_count

def all_dwd_single_code(code_name, icv_name, rclone_flag=True):
    """
    Calculates the total number of DWDs in the Galaxy (not just the LISA-
    detectable ones) for a single code/variation.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    icv_name: str
        Name of the initial condition variation (e.g. "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    total_dwd_count: float
        Total number of DWDs predicted in the Galaxy for that code/variation.
    """
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/initial_condition_variations/'
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/initial_condition_variations/'
    bin_data_filepath = initial_string + icv_name + '/' + code_name + \
        '_Galaxy_LISA_Candidates_Bin_Data.csv'
    
    bin_data_array = pd.read_csv(bin_data_filepath)
    total_dwd_count = sum(bin_data_array['SubBinNDWDsReal'])
    
    return total_dwd_count
