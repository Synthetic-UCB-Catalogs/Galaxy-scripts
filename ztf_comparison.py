import numpy as np
import os

ztf_array = np.genfromtxt('ZTF frequency distance.dat')
#the ZTF data used by the plots

def wd_radius_PPE(m):
    """
    Calculates radii of WDs based on mass. Called by other functions, no need
    to call this directly. Based on Verbunt & Rappaport (1988) with
    modifications from van Zeist et al. (2025).
    """
    
    # Eggleton 1986 fit to Nauenberg for high m and
    # ZS for low m. From Verbunt & Rappaport (1988)
    # 3/2 multiplier from van Zeist et al. (2025)
    
    
    fac1 = (m/1.44)**(2./3.)
    fac2 = 0.00057/m
    a = 3.5
    b = 1.

    r = (3/2)*0.0114*np.sqrt(1./fac1-fac1)*(1.+a*(fac2)**(2./3.)+b*fac2)**(-2./3.)
    #3/2 multiplier based on comparisons to ZTF masses and radii

    return r

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

def frequency_distance_bins(code_name, var_type, var_name, rclone_flag=True,
                            recalc_rad=True, ecl_weight=True):
    """
    Sorts the galaxy data into frequency/distance bins for the plotting code
    to use.
    Note: this uses the column ordering from the galaxy files from Katie's
    branch. Does not work with the column numbering from Alexey's branch. The
    line-by-line scanning would need to be tweaked in order to work dynamically
    with different column orderings.
    
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
    recalc_rad: bool
        If True, calculate radii based on the wd_radius_PPE() formula. If
        False, use whatever radii are provided in the galaxy file.
    ecl_weight: bool
        If True, weight systems by their eclipse probability when binning them.
    """
    
    """ Initialisation """
    
    log_freq_bin_bounds = np.linspace(-5,0,num=51)
    #every 0.1 dex in log-space: -5, -4.9, -4.8 etc.
    
    dist_upper_bound = 2000 #pc
    #dist_upper_bound_kpc = dist_upper_bound/1000
    dist_bin_bounds = np.linspace(0,dist_upper_bound,num=41) #every 50 pc, linearly
    
    amount_per_bin = np.zeros((50,40)) #zeros, not empty
    #50,40 instead of 51,41 because each bin is *between* the bounds from the lists above
    
    """ Fetching the right AllDWDs file """
    
    if var_type == 'icv' or var_type == 'initial_condition_variations':
        var_type_string = 'initial_condition_variations/'
        var_string = var_name
    elif var_type == 'mtv' or var_type == 'mass_transfer_variations':
        var_type_string = 'mass_transfer_variations/'
        #select appropriate subfolder in mass_transfer_variations
        if var_name == 'fiducial':
            var_string = var_name
        elif var_name == 'alpha_lambda_1' or var_name == 'alpha_lambda_2' or \
            var_name == 'alpha_lambda_02' or var_name == 'alpha_lambda_05' or \
            var_name == 'alpha_gamma_2':
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
    all_dwd_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_AllDWDs.csv'
        
    """ Running through the galaxy file """
    
    galaxy_file = open(all_dwd_filepath,'r')
    
    iteration_no = 0 #counter to keep track of progress
    iteration_kept = 0 #only those iterations kept within the distance limit
    
    for line in galaxy_file:
        if iteration_no == 0: #skip first row of file (headers)
            iteration_no += 1
            continue
        
        line_as_list = list(line.split(','))
        dist = float(line_as_list[33]) #pc
        
        if dist < dist_upper_bound:
            dist_bin = np.floor(dist/50) #bins are 50 pc (0.05 kpc) wide and start at 0 pc
        else: #skip systems at larger distances
            iteration_no += 1
            if (iteration_no + 1) % 1000000 == 0: print(str(iteration_no + 1) + ' systems done')
            continue
        
        m1 = float(line_as_list[7]) #Msun
        m2 = float(line_as_list[12]) #Msun
        a = float(line_as_list[4]) #Rsun
        
        if recalc_rad == True:
            r1 = wd_radius_PPE(m1) #Rsun
            r2 = wd_radius_PPE(m2) #Rsun
        else:
            r1 = float(line_as_list[8]) #Rsun
            r2 = float(line_as_list[13]) #Rsun
        
        freq = get_f_gw_from_semimajor(m1,m2,a) #Hz
        freq_bin = np.floor(10*np.log10(freq)) + 50
        #rounds to nearest 0.1, then adds 50 to map -5.0 (-50) to index 0
        
        if ecl_weight == True:
            system_weight = (r1 + r2)/a #eclipse probability; a and r in Rsun
        else:
            system_weight = 1
        
        #testing
        #print(str(r1) + '   ' + str(r2) + '  ' + str(freq_bin) + '  ' + str(dist_bin))
        #if iteration_no == 10:
        #    break
        
        amount_per_bin[int(freq_bin),int(dist_bin)] += system_weight
        #add the (weighted) system to the total for the appropriate freq/dist bin
        
        iteration_no += 1
        iteration_kept += 1
        if (iteration_no + 1) % 1000000 == 0: print(str(iteration_no + 1) + ' systems done') #+1 because of 0-indexing
        
    galaxy_file.close()
    
    print(iteration_no - 1) #-1 because first row skipped
    print(iteration_kept)
    print(sum(sum(amount_per_bin)))
    
    return amount_per_bin
