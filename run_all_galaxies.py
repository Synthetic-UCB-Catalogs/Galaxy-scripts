import galaxy
from pathlib import Path

def set_paths(ModelParams):
    '''Configure file paths based on ModelParams settings.
    
    Parameters:
    -----------
    ModelParams : dict
        Dictionary containing model parameters including 'RunWave', 'RunSubType', 'Code'
        
    Returns:
    --------
    T0_dat_path : str
        File path to the T0 data file.
    write_path : str
        Partial file path to save the Galaxy DataFrame.
    write_path_downsampled : str
        Partial file path to save the downsampled Galaxy DataFrame (if applicable).
    '''
    if ModelParams['RunWave'] == 'initial_condition_variations':
         if ModelParams['Code'] == 'SEVN_MIST':
             T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
         elif ModelParams['Code'] == 'BPASS':
             T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
         else:
             T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.hdf5'  # FilePath to the T0 data file

         if ModelParams['UseRepresentingWDs'] == True:
             write_path_downsampled = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType']   # Partial Filepath save the Galaxy DataFrame
         write_path = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType']   # Partial Filepath save the Galaxy DataFrame

    elif ModelParams['RunWave'] == 'mass_transfer_variations':
        var_name = ModelParams['RunSubType']
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

        if ModelParams['Code'] == 'SEVN_MIST':
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
        elif ModelParams['Code'] == 'BPASS':
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_T0.csv'  # FilePath to the T0 data file
        else:
            T0_dat_path = ModelParams['datPath'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string + '/' + ModelParams['Code'] + '_T0.hdf5'  # FilePath to the T0 data file

        if ModelParams['UseRepresentingWDs'] == True:
            write_path_downsampled = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons_lightweight_500K_DWDs/' + ModelParams['RunWave'] + '/' + var_string  # Partial Filepath save the Galaxy DataFrame
        write_path = ModelParams['datPath'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + var_string   # Partial Filepath save the Galaxy DataFrame
    else:
        raise ValueError('Invalid RunWave specified.')
    if not Path(write_path).exists():
        print(f'WARNING: write path does not exist: {write_path}')
        Path(write_path).mkdir(parents=True, exist_ok=True)

    if not Path(T0_dat_path).exists():
        print(f'WARNING: T0 path does not exist: {T0_dat_path}')
        return None, None, None
    
    return T0_dat_path, write_path, write_path_downsampled



def run_gx(RunWave, SubType, Code, datPath):
    ModelParams = { #Main options
                   'GalaxyModel': 'Besancon', #Currently can only be Besancon
                   'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used
                   'RecalculateCDFs': False, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
                   'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)
                   'RunWave': RunWave,
                   'RunSubType': SubType,
                   'Code': Code,
                   'ACutRSunPre': 6., #Initial cut for all DWD binaries
                   'UseRepresentingWDs': True, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, N, binaries
                   'RepresentDWDsBy': 500_000,  #Represent the present-day LISA candidates by this nubmer of binaries
                   'LISAPCutHours': (2/1.e-4)/(3600.), #LISA cut-off orbital period, 1.e-4 Hz + remember that GW frequency is 2X the orbital frequency
                   'MaxTDelay': 14000,
                   'DeltaTGalMyr': 50, #Time step resolution in the Galactic SFR
                   'UseOneBinOnly': False, #If False - use full model; if True - use just one bin, for visualization
                   'datPath': datPath
                   }
    
    T0_dat_path, write_path, write_path_downsampled = set_paths(ModelParams)
    
    if T0_dat_path is None:
        print('Skipping due to missing T0 data path.')
        print(f'check whether T0 path exists: {T0_dat_path}')
        print('')
        return None
    
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

    return None

def _run(args_in):
    RunWave, SubType, Code, datPath = args_in
    return run_gx(RunWave, SubType, Code, datPath)


if __name__ == '__main__':
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import argparse

    parser = argparse.ArgumentParser(description='Run Galaxy simulations across different configurations for simulated binary populations.')
    parser.add_argument('--datPath', type=str, required=True, help='Base path to the data directory containing simulated binary populations.')
    args = parser.parse_args()

    verbose=False
    RunWaves = ['initial_condition_variations', 'mass_transfer_variations']
    IC_subtypes = ['fiducial', 'thermal_ecc', 'uniform_ecc', 'm2_min_05', 'qmin_01', 'porb_log_uniform']
    MT_subtypes = ['alpha_lambda_1', 'alpha_lambda_2', 'alpha_lambda_02', 'alpha_lambda_05', 'alpha_gamma_2', 'qcrit_claeys_14',
                   'qcrit_hurley_02', 'qcrit_hurley_webbink', 'qcrit_zetas', 'accretion_0', 'accretion_1', 'accretion_05']
    #MT_subtypes = ['qcrit_claeys_14',
    #               'qcrit_hurley_02', 'qcrit_hurley_webbink', 'qcrit_zetas', 'accretion_0', 'accretion_1', 'accretion_05']
    
    #codes = ['COSMIC', 'BSE', 'SEVN_MIST', 'BPASS', 'SeBa', 'COMPAS', 'METISSE']
    codes = ['SEVN_MIST']
    
    dat_path = Path(args.datPath) / 'simulated_binary_populations' / 'monte_carlo_comparisons'
    
    missing_paths = []

    args_in = []
    for RW, subtype in zip(RunWaves, [IC_subtypes, MT_subtypes]):
        for s in subtype:
            for c in codes:
                if RW == 'initial_condition_variations':
                    path = dat_path / RW / s
                elif RW == 'mass_transfer_variations':
                    if s in ['alpha_lambda_1', 'alpha_lambda_2', 'alpha_lambda_02', 'alpha_lambda_05', 'alpha_gamma_2']:
                        path = dat_path / RW / 'common_envelope' / s
                    elif s in ['qcrit_claeys_14', 'qcrit_hurley_02', 'qcrit_hurley_webbink', 'qcrit_zetas']:
                        path = dat_path / RW / 'stability_of_mass_transfer' / s
                    elif s in ['accretion_0', 'accretion_1', 'accretion_05']:
                        path = dat_path / RW / 'stable_accretion_efficiency' / s
                    else:   
                        print(f'Unknown subtype: {s}')
                        continue
                if verbose:
                    print(f'trying: {RW}, {s}, {c}')

                if not path.exists():
                    print(f'Path does not exist: {path}')
                    print(f'Creating directory: {path}')
                    print('')
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                else:
                    if verbose:
                        print(f'loading data from {path}')
                args_in.append((RW, s, c, args.datPath))

    with Pool(cpu_count()) as pool:
        for _ in tqdm(
            pool.imap_unordered(_run, args_in),
            total=len(args_in),
        ):
            pass
    
    print('All done!')