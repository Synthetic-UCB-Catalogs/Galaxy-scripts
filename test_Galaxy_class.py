import galaxy
import os

# set the model parameters for the Galaxy class
ModelParams = { #Main options
               'GalaxyModel': 'Besancon', #Currently can only be Besancon
               'RecalculateNormConstants': True, #If true, density normalisations are recalculated and printed out, else already existing versions are used
               'RecalculateCDFs': True, #If true, the galaxy distribution CDFs are recalculated (use True when running first time on a new machine)
               'ImportSimulation': True, #If true, construct the present-day DWD populaiton (as opposed to the MS population)               
               #Simulation options
               'RunWave': 'initial_condition_variations',
               'RunSubType': 'fiducial',
               #'RunSubType': 'thermal_ecc',
               #'RunSubType': 'uniform_ecc',
               #'RunSubType': 'm2_min_05',
               #'RunSubType': 'qmin_01',
               #'RunSubType': 'porb_log_uniform',
               #'Code': 'COSMIC',
               #'Code': 'METISSE',
               'Code': 'SeBa',     
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
               'NPoints': 1e5 # Number of stars to sample if we just sample present-day stars
    }

T0_dat_path = os.environ['UCB_GOOGLE_DRIVE_DIR'] + '/simulated_binary_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code'] + '_T0.hdf5'  # FilePath to the T0 data file
write_path = os.environ['UCB_GOOGLE_DRIVE_DIR'] + '/simulated_galaxy_populations/monte_carlo_comparisons/' + ModelParams['RunWave'] + '/' + ModelParams['RunSubType'] + '/' + ModelParams['Code']  # Partial Filepath save the Galaxy DataFrame


# Import the Galaxy class from galaxy
gx = galaxy.Galaxy(ModelParams=ModelParams, T0_dat_path=T0_dat_path)

# Load possible LISA sources
try:
    gx.load_possible_LISA_sources()
except ValueError as ve:
    print(f"ValueError: {ve}")

gx.create_galaxy(write_path=write_path, verbose=False, write_h5=False)