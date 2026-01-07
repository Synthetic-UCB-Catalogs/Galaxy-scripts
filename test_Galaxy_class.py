import galaxy
import os
import argparse

# set the model parameters for the Galaxy class
ModelParams = { #Main options
               'GalaxyModel': 'Besancon', #Currently can only be Besancon
               'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'RecalculateCDFs': True, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)               
               #Simulation options
               'RunWave': 'initial_condition_variations',
               #'RunWave': 'mass_transfer_variations',
               'RunSubType': 'fiducial',
               #'RunSubType': 'thermal_ecc',
               #'RunSubType': 'uniform_ecc',
               #'RunSubType': 'm2_min_05',
               #'RunSubType': 'qmin_01',
               #'RunSubType': 'porb_log_uniform',
               #'RunSubType': 'accretion_1',
               #'Code': 'BPASS',
               'Code': 'BSE',
               #'Code': 'COSMIC',
               #'Code': 'METISSE',
               #'Code': 'SeBa',     
               #'Code': 'SEVN',
               #'Code': 'ComBinE',
               #'Code': 'COMPAS',
               #Simulation parameters
               'ACutRSunPre': 6., #Initial cut for all DWD binaries
               'UseRepresentingWDs': False, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, N, binaries
               'RepresentDWDsBy': 500000,  #Represent the present-day LISA candidates by this nubmer of binaries
               'LISAPCutHours': (2/1.e-4)/(3600.), #LISA cut-off orbital period, 1.e-4 Hz + remember that GW frequency is 2X the orbital frequency
               'MaxTDelay': 14000,
               'DeltaTGalMyr': 50, #Time step resolution in the Galactic SFR
               #Extra options
               'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualizations
               'OneBinToUse': 10, #Number of the bin, if only one bin in used
               'NPoints': 1e5, # Number of stars to sample if we just sample present-day stars
               'datPath': 'path_to_dat' #os.environ['UCB_GOOGLE_DRIVE_DIR'] # replace with path to your data
    }

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

if ModelParams['RunWave'] == 'initial_condition_variations':
    if ModelParams['Code'] == 'SEVN':
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_MIST_T0.csv'  # FilePath to the T0 data file
    elif ModelParams['Code'] == 'BPASS':
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
    else:
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.hdf5'  # FilePath to the T0 data file
    
    if ModelParams['UseRepresentingWDs'] == True:
        write_path_downsampled = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code']  # Partial Filepath save the Galaxy DataFrame
    write_path = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code']  # Partial Filepath save the Galaxy DataFrame

elif ModelParams['RunWave'] == 'mass_transfer_variations':
    var_name = ModelParams['RunSubType']
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
        raise ValueError('Invalid mass transfer variation specified in RunSubType.')
    
    if ModelParams['Code'] == 'SEVN':
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_MIST_T0.csv'  # FilePath to the T0 data file
    elif ModelParams['Code'] == 'BPASS':
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
    else:
        T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_T0.hdf5'  # FilePath to the T0 data file
    
    if ModelParams['UseRepresentingWDs'] == True:
        write_path_downsampled = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code']  # Partial Filepath save the Galaxy DataFrame
    write_path = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code']  # Partial Filepath save the Galaxy DataFrame
else:
    raise ValueError('Invalid RunWave specified.')


# Import the Galaxy class from galaxy
gx = galaxy.Galaxy(ModelParams=ModelParams, T0_dat_path=T0_dat_path)

# Load possible LISA sources
try:
    gx.load_possible_LISA_sources()
except ValueError as ve:
    print(f"ValueError: {ve}")

gx.create_galaxy(write_path=write_path, verbose=False, write_h5=False)
if ModelParams['UseRepresentingWDs']:
    gx.create_downsampled_galaxy(write_path=write_path_downsampled, verbose=False, write_h5=False)