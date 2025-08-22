import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def dwd_count_single_code(code_name, var_type, var_name, rclone_flag=True):
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
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_name: str
        Name of the initial condition/mass transfer variation (e.g.
        "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    dwd_count: float
        Number of LISA DWDs predicted in the Galaxy for that code/variation.
    """
    
    if var_type == 'icv' or var_type == 'initial_condition_variations':
        var_type_string = 'initial_condition_variations/'
        var_string = var_name
    elif var_type == 'mtv' or var_type == 'mass_transfer_variations':
        var_type_string = 'mass_transfer variations/'
        #select appropriate subfolder in mass_transfer_variations
        if var_name == 'fiducial':
            var_string = var_name
        elif var_name == 'alpha_lambda_1' or var_name == 'alpha_lambda_2' or \
            var_name == 'alpha_lambda_02' or var_name == 'alpha_lambda_05':
                var_string = 'common_envelope/' + var_name
        elif var_name == 'qcrit_claeys_14' or var_name == 'qcrit_hurley_02' \
            or var_name == 'qcrit_hurley_webbink' or var_name == 'qcrit_zetas':
                var_string = 'stability_of_mass_transfer/' + var_name
        elif var_name == 'accretion_0' or var_name == 'accretion_1' or \
            var_name == 'accretion_05':
                var_string = 'stable_accretion_efficiency/' + var_name
        else:
            raise ValueError('Invalid mass transfer variation specified.')
    else:
        raise ValueError('Please specify either initial condition or mass ' +
                        'transfer variations.')
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
    lisa_dwd_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_LISA_DWDs.csv'
    
    lisa_dwd_array = pd.read_csv(lisa_dwd_filepath)
    
    dwd_count = len(lisa_dwd_array)
    
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
        var_count[i] = dwd_count_single_code(code_name, 'icv', icv_names[i], \
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
        var_count[i] = dwd_count_single_code(code_name, 'icv', icv_names[i], \
                                             rclone_flag)
    
    min_dwd_count = np.min(var_count)
    max_dwd_count = np.min(var_count)
    
    return min_dwd_count, max_dwd_count

def all_dwd_single_code(code_name, var_type, var_name, rclone_flag=True):
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
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_name: str
        Name of the initial condition/mass transfer variation (e.g.
        "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    total_dwd_count: float
        Total number of DWDs predicted in the Galaxy for that code/variation.
    """
    
    if var_type == 'icv' or var_type == 'initial_condition_variations':
        var_type_string = 'initial_condition_variations/'
        var_string = var_name
    elif var_type == 'mtv' or var_type == 'mass_transfer_variations':
        var_type_string = 'mass_transfer variations/'
        #select appropriate subfolder in mass_transfer_variations
        if var_name == 'fiducial':
            var_string = var_name
        elif var_name == 'alpha_lambda_1' or var_name == 'alpha_lambda_2' or \
            var_name == 'alpha_lambda_02' or var_name == 'alpha_lambda_05':
                var_string = 'common_envelope/' + var_name
        elif var_name == 'qcrit_claeys_14' or var_name == 'qcrit_hurley_02' \
            or var_name == 'qcrit_hurley_webbink' or var_name == 'qcrit_zetas':
                var_string = 'stability_of_mass_transfer/' + var_name
        elif var_name == 'accretion_0' or var_name == 'accretion_1' or \
            var_name == 'accretion_05':
                var_string = 'stable_accretion_efficiency/' + var_name
        else:
            raise ValueError('Invalid mass transfer variation specified.')
    else:
        raise ValueError('Please specify either initial condition or mass ' +
                        'transfer variations.')
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
    bin_data_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_LISA_Candidates_Bin_Data.csv'
    
    bin_data_array = pd.read_csv(bin_data_filepath)
    total_dwd_count = sum(bin_data_array['SubBinNDWDsReal'])
    
    return total_dwd_count

def lisa_dwd_count_plotter(code_list, var_type, var_list, cmap='rainbow', \
                           rclone_flag=True):
    """
    Plots the number of LISA DWDs in the Galaxy for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_list: list of strs
        List of the names of the variations you want to plot.
    cmap: str
        Pyplot colormap to use for the bar plot. Defaults to 'rainbow', but we
        recommend 'gist_rainbow' if you are comparing many (5+) variations.
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
    """

    fig, ax = plt.subplots()
    width = 0.7/len(var_list) #make bars narrower if plotting more variations

    plot_colormap = plt.get_cmap(cmap)
    plot_colors = plot_colormap(np.linspace(0,1,len(var_list)))

    for i in range(len(code_list)):
        for j in range(len(var_list)):
            try: ax.bar(i+j*width, dwd_count_single_code(code_list[i], \
                 var_type, var_list[j], rclone_flag), width, \
                 color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
    
def total_dwd_count_plotter(code_list, var_type, var_list, cmap='rainbow', \
                           rclone_flag=True):
    """
    Plots the total number of DWDs in the Galaxy (not just the LISA-detectable
    ones) for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_list: list of strs
        List of the names of the variations you want to plot.
    cmap: str
        Pyplot colormap to use for the bar plot. Defaults to 'rainbow', but we
        recommend 'gist_rainbow' if you are comparing many (5+) variations.
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
    """

    fig, ax = plt.subplots()
    width = 0.7/len(var_list) #make bars narrower if plotting more variations

    plot_colormap = plt.get_cmap(cmap)
    plot_colors = plot_colormap(np.linspace(0,1,len(var_list)))

    for i in range(len(code_list)):
        for j in range(len(var_list)):
            try: ax.bar(i+j*width, all_dwd_single_code(code_list[i], \
                 var_type, var_list[j], rclone_flag), width, \
                 color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
