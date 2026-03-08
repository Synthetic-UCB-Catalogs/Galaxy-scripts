import galaxy
import utils

def run_gx(run_wave, run_sub_type, code, dat_path, run_full_galaxy=True, run_downsampled_galaxy=True, overwrite_path=None):
    ModelParams = { #Main options
                   'galaxy_model': 'Besancon', #Currently can only be Besancon
                   'recalculate_cdfs': False, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
                   'run_wave': run_wave,
                   'run_sub_type': run_sub_type,
                   'code': code,
                   'semiMajor_max': 6., #Initial cut for all DWD binaries
                   'create_downsampled_gx': run_downsampled_galaxy, #If False - each binary in the Galaxy is drawn as 1 to 1; if True - all the Galactic DWDs are represented by a smaller number, N, binaries
                   'downsample_fac': 10,  #Downsample the present-day LISA candidates by this factor
                   'f_LISA_low': 1e-4, #LISA lower GW freqency cut-off in Hz
                   'f_LISA_high': 1e-1, #LISA upper GW frequency cut-off in Hz
                   'age_max': 14000, #maximum age of the halo is 14 Gyr
                   'dat_path': dat_path,
                   'delta_t_gal_myr': 0.5, #Time step resolution in the Galactic SFR
                   'cols_write': ['ID', 'age', 'mass1', 'mass2', 'semiMajor_today', 'X_gx', 'Y_gx', 'Z_gx', 'dist', 'component'], #Columns to write to the galaxy output file
                   'midpoint': True
                   }
    
    T0_dat_path, write_path, write_path_downsampled = utils.set_paths(ModelParams)
    if overwrite_path is not None:
        write_path = overwrite_path
        write_path_downsampled = overwrite_path
        print(f'Overwriting default write paths with {write_path} and {write_path_downsampled} for full and downsampled galaxy, respectively.')

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

    if run_full_galaxy:
        gx.create_galaxy(write_path=write_path, verbose=False, write_h5=False, midpoint=ModelParams['midpoint'])
    if run_downsampled_galaxy:
        if write_path_downsampled is not None:
            gx.create_downsampled_galaxy(write_path=write_path_downsampled, verbose=False, write_h5=False)
        else:
            print('Skipping downsampled galaxy creation due to missing write path.')
            print(f'check whether downsampled write path exists: {write_path_downsampled}')
            print('')
    if not run_full_galaxy and not run_downsampled_galaxy:
        print('WARNING: neither full galaxy nor downsampled galaxy will be created since both run_full_galaxy and run_downsampled_galaxy are set to False.')    


    return None

def _run(args_in):
    run_wave, SubType, code, dat_path, run_full_galaxy, run_downsampled_galaxy, overwrite_path = args_in
    return run_gx(run_wave, SubType, code, dat_path, run_full_galaxy, run_downsampled_galaxy, overwrite_path=overwrite_path)


if __name__ == '__main__':
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Run Galaxy simulations across different configurations for simulated binary populations.')
    parser.add_argument('--dat_path', type=str, required=True, help='Base path to the data directory containing simulated binary populations.')
    parser.add_argument('--overwrite_path', type=str, required=False, help='Path to overwrite the default output path for the galaxy data.')

    args = parser.parse_args()

    multi = True
    verbose=True
    run_full_galaxy = True
    run_downsampled_galaxy = False
    #run_waves = ['initial_condition_variations', 'mass_transfer_variations']
    #initial_conditions_variations = ['fiducial', 'thermal_ecc', 'uniform_ecc', 
    #                                 'm2_min_05', 'qmin_01', 'porb_log_uniform']
    #mass_transfer_variations = ['alpha_lambda_1', 'alpha_lambda_2', 'alpha_lambda_02', 'alpha_lambda_05', 
    #                            'alpha_gamma_2', 'qcrit_claeys_14','qcrit_hurley_02', 'qcrit_hurley_webbink', 
    #                            'qcrit_zetas', 'accretion_0', 'accretion_1', 'accretion_05']
    
    run_waves = ['initial_condition_variations']
    initial_conditions_variations = ['fiducial']
    
    #codes = ['COSMIC', 'BSE', 'SEVN_MIST', 'BPASS', 'SeBa', 'COMPAS', 'METISSE']
    #codes = ['COSMIC', 'BSE', 'SeBa']
    codes = ['COSMIC']
    
    dat_path = Path(args.dat_path) / 'simulated_binary_populations' / 'monte_carlo_comparisons'
    
    missing_paths = []
    args_in = []
    for rw, subtype in zip(run_waves, [initial_conditions_variations]):
        for s in subtype:
            for c in codes:
                if rw == 'initial_condition_variations':
                    path = dat_path / rw / s
                elif rw == 'mass_transfer_variations':
                    if s in ['alpha_lambda_1', 'alpha_lambda_2', 'alpha_lambda_02', 'alpha_lambda_05', 'alpha_gamma_2']:
                        path = dat_path / rw / 'common_envelope' / s
                    elif s in ['qcrit_claeys_14', 'qcrit_hurley_02', 'qcrit_hurley_webbink', 'qcrit_zetas']:
                        path = dat_path / rw / 'stability_of_mass_transfer' / s
                    elif s in ['accretion_0', 'accretion_1', 'accretion_05']:
                        path = dat_path / rw / 'stable_accretion_efficiency' / s
                    else:   
                        print(f'Unknown subtype: {s}')
                        continue

                if not path.exists():
                    print(f'Path does not exist: {path}')
                    print(f'Creating directory: {path}')
                    print('')
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                
                args_in.append((rw, s, c, args.dat_path, run_full_galaxy, run_downsampled_galaxy, args.overwrite_path))

    if multi:
        with Pool(cpu_count()) as pool:
            for _ in tqdm(
                pool.imap_unordered(_run, args_in),
                total=len(args_in),
            ):
                pass
    else:
        for arg in tqdm(args_in):
            _run(arg)
    print('All done!')