import pandas as pd

def dwd_count_single_code(code_name, icv_name):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code/variation. Filepaths assume you have the top-level directory in the
    project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    icv_name: str
        Name of the initial condition variation (e.g. "fiducial").
    """
    
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
