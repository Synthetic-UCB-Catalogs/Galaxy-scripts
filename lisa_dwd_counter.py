import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    lisa_dwd_filepath = initial_string + icv_name + '/' + code_name + \
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

def lisa_dwd_count_plotter(code_list, var_list, cmap='rainbow', \
                           rclone_flag=True):
    """
    Plots the number of LISA DWDs in the Galaxy for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_list: list of strs
        List of the names of the variations you want to plot. Currently
        supports only initial conditions variations.
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
                 var_list[j], rclone_flag), width, color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
    
def total_dwd_count_plotter(code_list, var_list, cmap='rainbow', \
                           rclone_flag=True):
    """
    Plots the total number of DWDs in the Galaxy (not just the LISA-detectable
    ones) for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_list: list of strs
        List of the names of the variations you want to plot. Currently
        supports only initial conditions variations.
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
                 var_list[j], rclone_flag), width, color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
