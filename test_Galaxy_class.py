import galaxy
import os
import argparse

# set the model parameters for the Galaxy class
ModelParams = { #Main options
               'GalaxyModel': 'Besancon', #Currently can only be Besancon
               'RecalculateNormConstants': False, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'recalculate_cdfs': False, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)               
               #Simulation options
               'run_wave': 'initial_condition_variations',
               #'RunWave': 'mass_transfer_variations',
               'RunSubType': 'fiducial',
               #'RunSubType': 'thermal_ecc',
               #'RunSubType': 'uniform_ecc',
               #'RunSubType': 'm2_min_05',
               #'RunSubType': 'qmin_01',
               #'RunSubType': 'porb_log_uniform',
               #'RunSubType': 'accretion_1',
               #'code': 'BPASS',
               #'code': 'BSE',
               'code': 'COSMIC',
               #'code': 'METISSE',
               #'code': 'SeBa',     
               #'code': 'SEVN',
               #'code': 'ComBinE',
               #'code': 'COMPAS',
               #Simulation parameters
               'semiMajor_max': 6., #Initial cut for all DWD binaries
               'create_downsampled_gx': False, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, N, binaries
               'downsample_fac': 10,  #Downsample the present-day LISA candidates by this factor
               'f_LISA_low': 1e-4, #LISA lower GW freqency cut-off in Hz
               'f_LISA_high': 1e-1, #LISA upper GW frequency cut-off in Hz
               'age_max': 14000, #maximum age of the halo is 14 Gyr
               'dat_path': dat_path
               #'datPath': os.environ['UCB_GOOGLE_DRIVE_DIR'] #use if rclone is set up
    }





if __name__ == '__main__':
    # Use argparse to optionally override ModelParams from command line
    parser = argparse.ArgumentParser(description='Set ModelParams for Galaxy class.')
    for key, item in ModelParams.items():
        if isinstance(item, bool):
            parser.add_argument(f'--{key}', type=lambda x: (str(x).lower() == 'true'), default=item, help=f'Set {key} (default: {item})')
        else:
            parser.add_argument(f'--{key}', type=type(item), default=item, help=f'Set {key} (default: {item})')
    
    args = parser.parse_args()
    for key in ModelParams.keys():
        ModelParams[key] = getattr(args, key)
    
    if ModelParams['run_wave'] == 'initial_condition_variations':
        if ModelParams['code'] == 'SEVN':
            T0_dat_path = ModelParams['dat_path'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['run_wave'] + '/' + ModelParams['run_sub_type'] + '/' + ModelParams['code'] + '_MIST_T0.csv'  # FilePath to the T0 data file
        elif ModelParams['code'] == 'BPASS':
            T0_dat_path = ModelParams['dat_path'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['run_wave'] + '/' + ModelParams['run_sub_type'] + '/' + ModelParams['code'] + '_T0.csv'  # FilePath to the T0 data file
        else:
            T0_dat_path = ModelParams['dat_path'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['run_wave'] + '/' + ModelParams['run_sub_type'] + '/' + ModelParams['code'] + '_T0.hdf5'  # FilePath to the T0 data file
        
        if ModelParams['UseRepresentingWDs'] == True:
            write_path_downsampled = ModelParams['dat_path'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['run_wave'] + '/' + ModelParams['run_sub_type'] + '/' + ModelParams['code']  # Partial Filepath save the Galaxy DataFrame
        write_path = ModelParams['dat_path'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['run_wave'] + '/' + ModelParams['run_sub_type'] + '/' + ModelParams['code']  # Partial Filepath save the Galaxy DataFrame
    
    elif ModelParams['run_wave'] == 'mass_transfer_variations':
        var_name = ModelParams['run_sub_type']
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
            raise ValueError('Invalid mass transfer variation specified in RunSubType.')
        
        if ModelParams['code'] == 'SEVN_MIST':
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['code'] + '_MIST_T0.csv'  # FilePath to the T0 data file
        elif ModelParams['code'] == 'BPASS':
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['code'] + '_T0.csv'  # FilePath to the T0 data file
        else:
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['code'] + '_T0.hdf5'  # FilePath to the T0 data file
        
        if ModelParams['UseRepresentingWDs'] == True:
            write_path_downsampled = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['code']  # Partial Filepath save the Galaxy DataFrame
        write_path = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['code']  # Partial Filepath save the Galaxy DataFrame
    else:
        raise ValueError('Invalid RunWave specified.')
    


# Import the Galaxy class from galaxy
gx = galaxy.Galaxy(ModelParams=ModelParams, T0_dat_path=T0_dat_path)

# Load possible LISA sources
try:
    gx.load_possible_LISA_sources()
except ValueError as ve:
    print(f"ValueError: {ve}")

if ModelParams['UseRepresentingWDs']:
    gx.create_downsampled_galaxy(write_path=write_path_downsampled, verbose=False, write_h5=False)
else:
    gx.create_galaxy(write_path=write_path, verbose=False, write_h5=False)
