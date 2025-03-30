import os
from .rapid_code_load_T0 import load_T0_data
from get_mass_norm import get_mass_norm

# This environment variable must be set by the user to point to the root of the google drive folder
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']


def getSimulationProperties(relative_path=None, output_dir=None):
    simulation_path = os.path.join(SIM_DIR, relative_path)
    if not os.path.isfile(simulation_path):
        raise Exception("File not found: {}".format(simulation_path))

    alldirs = simulation_path.split('/')
    RunWave = alldirs[-2]
    Code = alldirs[-1][:-8]  # remove the _T0.hdf5 to get the code name

    if output_dir is None:
        output_dir = './output/'
        # CurrOutDir      = './ProcessedSimulations/' + OutputSubfolder + '/' + RunWave + '/'
    os.makedirs(output_dir, exist_ok=True)

    df, header = load_T0_data(simulation_path)
    # print(df)
    # print(df.keys())
    DWDs_all = df.loc[(df.type1.isin([21, 22, 23])) & (df.type2.isin([21, 22, 23])) & (
        df.semiMajor > 0)].groupby('ID', as_index=False).first()  # Intact DWD binaries
    # print(DWDs_all)

    return DWDs_all, header

    # To sort out later:

    # Sims                          = SimData[0]
    # if not (Code == 'ComBinE'):
    #    DWDSetPre                     = df.loc[(df.type1.isin([21,22,23])) & (df.type2.isin([21,22,23])) & (df.semiMajor > 0) & (df.semiMajor < ACutRSunPre)].groupby('ID', as_index=False).first() #DWD binaries at the moment of formation with a<6RSun
    # else:
    #    # RTW TODO: why is it different for ComBinE? shouldn't all the codes be uniform under T0?
    #    DWDSetPre                     = df.loc[(df.type1 == 2) & (df.type2 == 2) & (df.semiMajor > 0) & (df.semiMajor < ACutRSunPre)].groupby('ID', as_index=False).first() #DWD binaries at the moment of formation with a<8RSun

    #
    # Import data
    # Parameters
    # RunWave         = 'fiducial'
    # RunWave         = 'porb_log_uniform'
    # RunWave         = 'uniform_ecc'
    # RunWave         = 'qmin_01'
    # Code            = 'COSMIC'
    # Code            = 'ComBinE'
    # Code            = 'COMPAS'
    # FileName        = './Simulations/' + RunWave + '/' + Code + '_T0.hdf5'
    # OutputSubfolder = 'ICVariations'
    # CurrOutDir      = './ProcessedSimulations/' + OutputSubfolder + '/' + RunWave + '/'

    # ACutRSunPre     = 6     #Initial cut for all DWD binaries
    # LISAPCutHours   = (2/1.e-4)/(3600.)  #1/e-4 Hz + remember that GW frequency is 2X the orbital frequency
    # MaxTDelay       = 14000
    # RepresentDWDsBy = 500000     #Represent the present-day LISA candidates by this nubmer of binaries
    # DeltaTGalMyr    = 50         #Time step resolution in the Galactic SFR
    #
    # General quantities
    # MassNorm        = get_mass_norm(RunWave)
    # NStarsPerRun    = GalaxyParams['MGal']/MassNorm
    # NRuns           = SimData[1]['NSYS'][0]
    #
    # Pre-process simulations
    # General properties
    # Lower-mass WD radius
    # DWDSetPre['RDonorRSun']       = RWD(np.minimum(DWDSetPre['mass1'],DWDSetPre['mass2']))
    # Mass ratio (lower-mass WD mass/higher-mass WD mass)
    # DWDSetPre['qSet']             = np.minimum(DWDSetPre['mass1'],DWDSetPre['mass2'])/np.maximum(DWDSetPre['mass1'],DWDSetPre['mass2'])
    # RLO separation for the lower-mass WD
    # DWDSetPre['aRLORSun']         = DWDSetPre['RDonorRSun']/fRL(DWDSetPre['qSet'])
    # Period at DWD formation
    # DWDSetPre['PSetDWDFormHours'] = POrbYr(DWDSetPre['mass1'],DWDSetPre['mass2'], DWDSetPre['semiMajor'])*YearToSec/(3600.)
    # Period at RLO
    # DWDSetPre['PSetRLOHours']     = POrbYr(DWDSetPre['mass1'],DWDSetPre['mass2'], DWDSetPre['aRLORSun'])*YearToSec/(3600.)
    #
    # GW-related timescales
    # Point mass GW inspiral time from DWD formation to zero separation
    # DWDSetPre['TGWMyrSetTot']            = TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['semiMajor'])
    # Point mass GW inspiral time from DWD formation to LISA band (or zero, if we are in the band already)
    # DWDSetPre['aLISABandRSun']           = AComponentRSun(DWDSetPre['mass1'], DWDSetPre['mass2'], (LISAPCutHours*3600)/YearToSec)
    # DWDSetPre['TGWMyrToLISABandSet']     = (DWDSetPre['TGWMyrSetTot'] - TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['aLISABandRSun'])).clip(0)
    # Point mass GW inspiral time from LISA band (or current location if we are in the band) to RLO
    # DWDSetPre['TGWMyrLISABandToRLOSet']  = TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],np.minimum(DWDSetPre['aLISABandRSun'],DWDSetPre['semiMajor'])) - TGWMyr(DWDSetPre['mass1'],DWDSetPre['mass2'],DWDSetPre['aRLORSun'])
    # Time from DMS formation to the DWD entering the LISA band
    # DWDSetPre['AbsTimeToLISAMyr']        = DWDSetPre['time'] + DWDSetPre['TGWMyrToLISABandSet']
    # Time from DMS formation to the DWD RLO
    # DWDSetPre['AbsTimeToLISAEndMyr']     = DWDSetPre['AbsTimeToLISAMyr'] + DWDSetPre['TGWMyrLISABandToRLOSet']
    #
    # Select DWDs that: 1)Do not merge upon formation, 2) Will reach the LISA band within the age of the Universe
    # DWDSet                       = DWDSetPre.loc[(DWDSetPre.semiMajor > DWDSetPre.aRLORSun) & (DWDSetPre.AbsTimeToLISAMyr < MaxTDelay)].sort_values('AbsTimeToLISAMyr')
    # Total number of DWDs produced in the simulation
    # NDWDLISAAllTimesCode         = len(DWDSet.index)
    # Corresponding total number of DWDs ever formed in the MW
    # NDWDLISAAllTimesReal         = NDWDLISAAllTimesCode*NStarsPerRun
    #
    #
    # Get the number of present-day potential LISA sources
    # Track the considered sub-bin
    # SubComponentCounter    = 0
    # Make a DF that tracks DWD counts, times etc in each sub-bin
    # SubComponentProps      = []
    # Make a dict that keeps DWD ID pointers for each sub-bin
    # SubComponentDWDIDDict     = {}
    # Go over each Besancon bin
    # for iComponent in range(len(BesanconParamsDefined['ComponentName'])):
    #    print('Step 1: ', iComponent, '/',(len(BesanconParamsDefined['ComponentName'])))
    #    #Component start and end times
    #    TGalComponentStart = BesanconParamsDefined['AgeMin'][iComponent]
    #    TGalComponentEnd   = BesanconParamsDefined['AgeMax'][iComponent]
    #    #Galactic mass fraction in the bin
    #    GalComponentProb   = NormConstantsDict['ComponentMassFractions'][iComponent]
    #    #Number of sub-bins, equally spaced in time; one sub-bin for starburst bins
    #    NSubComponents     = int(np.floor((TGalComponentEnd - TGalComponentStart)/DeltaTGalMyr) + 1)
    #    #Time duration of each sub-bin in this bin
    #    CurrDeltaT   = (TGalComponentEnd - TGalComponentStart)/NSubComponents
    #    #Galactic mass fraction per sub-bin
    #    GalSubComponentProb = GalComponentProb/NSubComponents
    #    #Initialise the start and end time of the current sub-bin
    #    CurrTMin      = TGalComponentStart
    #    CurrTMax      = TGalComponentStart + CurrDeltaT
    #    #print('Component:', iComponent)
    #    #Loop over sub-bins
    #    for jSubComponent in range(NSubComponents):
    #        #Mid-point in time
    #        CurrTMid            = 0.5*(CurrTMin + CurrTMax)
    #        #Current LISA sources (formed before today, will leave the band after today)
    #        LISASourcesCurrDF   = DWDSet[(DWDSet['AbsTimeToLISAMyr'] < CurrTMid) & (DWDSet['AbsTimeToLISAEndMyr'] > CurrTMid)]
    #        #The expected number of LISA sources,
    #        CurrSubComponentNDWDsCode = len(LISASourcesCurrDF.index)
    #        CurrSubComponentDWDReal   = GalSubComponentProb*NStarsPerRun*CurrSubComponentNDWDsCode
    #        #Log sub-bin properties
    #        SubComponentProps.append({'SubComponentAbsID':SubComponentCounter, 'SubComponentLocalID':jSubComponent, 'ComponentID':iComponent, 'SubComponentMidAge': CurrTMid, 'SubComponentDeltaT': CurrDeltaT,
    #                            'SubComponentNDWDsCode': CurrSubComponentNDWDsCode,
    #                            'SubComponentNDWDsReal': CurrSubComponentDWDReal})
    #        #Log DWDs
    #        SubComponentDWDIDDict[SubComponentCounter] = LISASourcesCurrDF
    #        #print(CurrTMin, CurrTMax, NLISASourcesCurr)
    #        SubComponentCounter += 1
    #        CurrTMin      += CurrDeltaT
    #        CurrTMax      += CurrDeltaT
    #        #print(NSubComponents)

    # Make a DF for the present-day population properties
    # SubComponentDF = pd.DataFrame(SubComponentProps)
    # Export the population properties

    # SubComponentDF.to_csv(CurrOutDir + Code + '_Galaxy_LISA_Candidates_Component_Data.csv', index = False)
    #
    # Get overall present-day properties
    # Total real number of LISA sources
    # NLISACandidatesToday           = np.sum(SubComponentDF['SubComponentNDWDsReal'])
    # Total number of simulations available to draw from
    # NLISACandidatesTodaySimulated  = np.sum(SubComponentDF['SubComponentNDWDsCode'])
    # Fraction of DWDs that have formed and become present-day LISA sources
    # FracLISADWDsfromAllDWDs        = NLISACandidatesToday/NDWDLISAAllTimesReal
    # What fraction of the needed DWDs we have simulated (approximate number)
    # FracSimulated                  = NDWDLISAAllTimesCode/NLISACandidatesToday
    #
    # Auxiliary function to make rounding statistically equal to averaged
    # def probabilistic_round(N):
    #    lower = int(N)
    #    upper = lower + 1
    #    fractional_part = N - lower
    #    return upper if random.random() < fractional_part else lower
    #
    #
    # Make a dataset of the present-day LISA DWD candidates
    # Draw the number of objects from each sub-bin in proportion to the number of real DWD LISA candidates expected from this sub-bin
    # NFindPre     = RepresentDWDsBy
    # NFindSubComponents = np.array([probabilistic_round((NFindPre/NLISACandidatesToday)*SubComponentDF['SubComponentNDWDsReal'].iloc[i]) for i in range(SubComponentCounter)],dtype=int)
    # NFind        = np.sum(NFindSubComponents)
    # PresentDayDWDCandFinSet    = []
    # Do the actual drawing
    # for iSubComponent in range(SubComponentCounter):
    #    CurrFind     = NFindSubComponents[iSubComponent]
    #    if CurrFind > 0:
    #        PresentDayDWDCandFin   = SubComponentDWDIDDict[iSubComponent].sample(n=CurrFind, replace=True)
    #        SubComponentRow              = SubComponentDF.iloc[iSubComponent]
    #        SubComponentData             = pd.DataFrame([SubComponentRow.values] * len(PresentDayDWDCandFin), columns=SubComponentRow.index)
    #        PresentDayDWDCandFinSet.append(pd.concat([PresentDayDWDCandFin.reset_index(drop=True), SubComponentData.reset_index(drop=True)], axis=1))
    #
    # PresentDayDWDCandFinDF = pd.concat(PresentDayDWDCandFinSet, ignore_index=True)
    #
    # Find present-day periods:
    # PresentDayDWDCandFinDF['ATodayRSun']     = APostGWRSun(PresentDayDWDCandFinDF['mass1'], PresentDayDWDCandFinDF['mass2'], PresentDayDWDCandFinDF['semiMajor'], PresentDayDWDCandFinDF['SubComponentMidAge'] - PresentDayDWDCandFinDF['time'])
    # PresentDayDWDCandFinDF['PSetTodayHours'] = POrbYr(PresentDayDWDCandFinDF['mass1'],PresentDayDWDCandFinDF['mass2'], PresentDayDWDCandFinDF['ATodayRSun'])*YearToSec/(3600.)


if __name__ == "__main__":
    relative_path = "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COMPAS_T0.hdf5"
    getSimulationProperties(relative_path=relative_path)
