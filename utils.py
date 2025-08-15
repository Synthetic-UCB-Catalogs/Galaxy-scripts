import numpy as np
import h5py
def get_bin_frac_ratio(IC_model, binary_fraction=0.5):
    from scipy.interpolate import CubicSpline

    # these are the hard coded ratios based on initial conditions sampling
    # tests done by K. Breivik using binary fractions from 0.1-1.0
    ratio_dict = {
        'm2_min_05': [5.497065159607499, 4.028385607219321, 3.1320443249502383, 2.541669993010094,
                      2.105948317768266, 1.786314805728416, 1.5164506548687628, 1.313816074228385,
                      1.141178996587137, 0.9968331286953663, 0.8730241509546721, 0.7696547388705937,
                      0.6788737987994921, 0.5998311488995663, 0.5308139080430776, 0.4699942084336498,
                      0.41210469109378484, 0.36061186819185237, 0.31590620434046357, 0.27425387234835646,
                      0.23543985197379194, 0.19987998623485503, 0.16926018076106297, 0.13969418254623414,
                      0.11143821417288673, 0.08563401307212755, 0.06239655229658728, 0.03973086411370632,
                      0.019317445609880482, 0.0],
        'porb_log_uniform': [5.886862859310299, 4.3420507204072525, 3.405093627693288, 2.7466102648326007,
                             2.27052928301355, 1.9273925451530112, 1.6405699036728623, 1.4078089101945124,
                             1.2331941437437164, 1.069552690381338, 0.9456643329362575, 0.8327939389648276,
                             0.7359840090556246, 0.6480801205088771, 0.5722387172813348, 0.5039632125801372,
                             0.4440067853899282, 0.3913613364579611, 0.3402636518883438, 0.29843507961701765,
                             0.2543196815715736, 0.21741447251907284, 0.18304208420964435, 0.15107685764473266,
                             0.12041807829958329, 0.09269327057199453, 0.06744511227698262, 0.04274378038546733,
                             0.02137711323393566, 0.0],
        'uniform_ecc': [5.889211849314321, 4.361889407239361, 3.4020828542097057, 2.746002579297502,
                        2.2673692464054183, 1.9141940732871088, 1.6405959517007267, 1.4094329827706373,
                        1.230470758764182, 1.072042568070819, 0.943431196191181, 0.8333833246264916,
                        0.7309333535548196, 0.6475817013002746, 0.574417702589979, 0.5056945660498336,
                        0.44488117789149545, 0.3884654642073608, 0.3388900220256573, 0.2962626999212119,
                        0.2557025971124346, 0.21685490011138606, 0.18305848181583317, 0.15006625511716296,
                        0.12090585737779058, 0.09221860126908878, 0.06728132687768605, 0.043606002397485674,
                        0.021239106748386388, 0.0],
        'qmin_01': [5.823149995795932, 4.257871067861341, 3.3350623602831586, 2.701648948025014,
                    2.241763868198779, 1.8890779951715042, 1.6099148365641083, 1.3953172844589194,
                    1.2110201528032853, 1.0592751609913709, 0.9225325052920952, 0.8191978467059325,
                    0.7180939238557436, 0.6326983982513528, 0.5640129919280746, 0.49581103805511545,
                    0.43644888054132813, 0.3834832365660745, 0.3356910316544453, 0.29248525094604144,
                    0.2511267079306599, 0.21240840773299557, 0.1776591271238454, 0.14765505334148785,
                    0.11783814018942763, 0.09097678374483624, 0.06650626075146698, 0.04210676407248778,
                    0.0203586397778618, 0.0],
        'fiducial': [5.932296012899532, 4.357823767133379, 3.394145723221763, 2.743531290685849,
                     2.261347038645783, 1.9173522895783706, 1.6465752814658685, 1.409849902360307,
                     1.232375306106761, 1.0745464651773011, 0.9431907171044254, 0.8331178361129032,
                     0.7317070852105148, 0.6506914750851542, 0.5707015794152772, 0.5048166662383198,
                     0.44379738107168043, 0.3913719830983486, 0.33791478738462305, 0.29476240868778614,
                     0.2551125640746029, 0.21752011832877077, 0.18147403979103155, 0.14971671211685017,
                     0.1209730374015313, 0.09372046390290077, 0.06743095018359142, 0.043359831467175966,
                     0.02054866077700454, 0.0],
        'thermal_ecc': [5.9200035744176125, 4.341656072119606, 3.4004867453530436, 2.747854677340573,
                        2.2726296004773694, 1.9140533603622882, 1.6388693872876943, 1.4154780593615193,
                        1.2294748612513091, 1.0760766054593665, 0.9442728392343459, 0.8321788961090938,
                        0.7330076399499459, 0.6473674909796342, 0.5733259021802497, 0.5031744591982074,
                        0.44384269602503534, 0.38875209683437956, 0.3429980155840034, 0.2955812055263655,
                        0.25491188680860966, 0.21846399667917327, 0.18130416084339843, 0.14918121603752904,
                        0.12215748474240215, 0.09270863580218562, 0.06716028032728252, 0.04318939158189473,
                        0.020791845074920087, 0.0]
    }
    binfracs = np.linspace(0.1, 1.0, 30)

    # select the list of ratios based on the initial conditions model
    ratio = ratio_dict[IC_model]

    # set up a spline to get the ratio for any binfrac
    r_spline = CubicSpline(binfracs, ratio)

    return r_spline(binary_fraction)


def get_mass_norm(IC_model, binary_fraction=0.5):
    '''selects the mass normalization for the 
    initial conditions sample set based on 
    the IC_model name and a binary fraction

    Parameters
    ----------
    IC_model : `str`
        initial conditions model chosen from:
            ecc_uniform, ecc_thermal, porb_log_uniform, m2_min_05, qmin_01, fiducial

    Returns
    -------
    mass_norm : `float`
        the total ZAMS mass of the initial stellar population
        including single and binary stars
    '''

    mass_binaries = {
        'uniform_ecc': 2720671.1164002735,
        'thermal_ecc': 2700943.07050043,
        'porb_log_uniform': 2713046.6197530716,
        'm2_min_05': 2905718.830512573,
        'qmin_01': 5510313.245766795,
        'fiducial': 2697557.2681495477
    }

    # get the ratio of singles to binaries for the selected binary fraction
    ratio = get_bin_frac_ratio(IC_model, binary_fraction=binary_fraction)
    mass_total = mass_binaries[IC_model] * (1 + ratio)

    return mass_total


def galaxy_params(key):
    '''Returns the parameters for the Galaxy based on the key provided.
    Parameters
    ----------
    key : str
        The key for the parameter to retrieve. Possible keys include:
        'MGal', 'MBulge', 'MBulge2', 'MHalo', 'RGalSun', 'ZGalSun'.

    Returns
    ------- 
    float or None
        The requested parameter value, or None if the key is not found.
    '''

    GalaxyParams = {'MGal': 6.43e10, #From Licquia and Newman 2015
                'MBulge': 6.1e9, #From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8, #From Robin+ 2012, metal-poor bulge
                'MHalo': 1.4e9, #From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'RGalSun': 8.2, #Bland-Hawthorn, Gerhard 2016
                'ZGalSun': 0.025 #Bland-Hawthorn, Gerhard 2016
               }
    
    return GalaxyParams.get(key, None)


def Besancon_params(key):
   '''Returns the parameters for the Besancon model based on the key provided.
   Parameters
   ----------
   key : str
       The key for the parameter to retrieve. Possible keys include:
       'BinName', 'AgeMin', 'AgeMax', 'XRange', 'YRange', 'RRange',
       'ZRange', 'RNPoints', 'ZNPoints', 'FeHMean', 'FeHStD',
       'Rho0ParamSetMSunPcM3', 'SigmaWKmS', 'EpsSetThin', 'EpsHalo',
       'dFedR', 'CSVNames'.
   Returns
   -------
   np.ndarray or None
       The requested parameter as a numpy array, or None if the key is not found.
   '''
   
   BesanconParams = {
    'BinName':np.array(['ThinDisk1', 'ThinDisk2','ThinDisk3','ThinDisk4','ThinDisk5','ThinDisk6','ThinDisk7','ThickDisk','Halo','Bulge']),
    'AgeMin': 1000.*np.array([0,0.15,1,2,3,5,7,10,14,8],dtype='float64'),
    'AgeMax': 1000.*np.array([0.15,1.,2.,3.,5.,7.,10.,10.,14.,10],dtype='float64'),
    'XRange': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
    'YRange': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
    'RRange': np.array([30,30,30,30,30,30,30,30,50,5],dtype='float64'),
    'ZRange': np.array([4,4,4,4,4,4,4,8,50,3],dtype='float64'),
    'RNPoints': np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,500],dtype='int64'),
    'ZNPoints': np.array([800,800,800,800,800,800,800,800,800,400],dtype='int64'),
    'FeHMean': np.array([0.01,0.03,0.03,0.01,-0.07,-0.14,-0.37,-0.78,-1.78,0.00],dtype='float64'),
    'FeHStD': np.array([0.12,0.12,0.10,0.11,0.18,0.17,0.20,0.30,0.50,0.40],dtype='float64'),
    'Rho0ParamSetMSunPcM3': np.array([1.888e-3,5.04e-3,4.11e-3,2.84e-3,4.88e-3,5.02e-3,9.32e-3,2.91e-3,9.2e-6],dtype='float64'), #Czekaj2014
    'SigmaWKmS': np.array([6,8,10,13.2,15.8,17.4,17.5],dtype='float64'),
    'EpsSetThin': np.array([0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.0785, 0.0791],dtype='float64'),
    'EpsHalo': np.array([0.76],dtype='float64'),
    'dFedR': np.array([-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,-0.07,0,0,0],dtype='float64'),
    'CSVNames':np.array(['GalTestThin1.csv','GalTestThin2.csv','GalTestThin3.csv','GalTestThin4.csv','GalTestThin5.csv','GalTestThin6.csv','GalTestThin7.csv','GalTestThick.csv','GalTestHalo.csv','GalTestBulge.csv']),
    'NormCSet': [75107132.47035658, 338372651.3889757, 272616301.5304108, 187101048.78214946, 320793654.3767256, 329736969.56415313, 612154207.0001729, 7357543.467451542, 47008.07057443722, 1200277748.5793166],
    'BinMasses' : [573793912.9393324, 2988609452.097608, 3369164425.0053015, 3397558456.9258327, 7358242977.7614, 8530537681.592255, 15957925274.598352, 14364167819.07992, 1400000000.0, 6360000000.0],
    'BinMassFractions': [0.00892370004571279, 0.04647915166559266, 0.05239758048219754, 0.05283916729278122, 0.11443612718135926, 0.13266777109785777, 0.24817924221770377, 0.2233929676373238, 0.02177293934681182, 0.09891135303265941],
    'Alpha': 78.9 * (np.pi / 180),  # Robin+2012
    'Beta': 3.6 * (np.pi / 180),   # Robin+2012
    'Gamma': 91.3 * (np.pi / 180)  # Robin+2012
    }
   
   return BesanconParams.get(key, None)

   
#Routine to load data from a 2D-organised hdf5 file
def load_RZdicts_from_hdf5(file_path):
    ZCDFDictSet = {}
    
    # Open the file for reading
    with h5py.File(file_path, 'r') as hdf5_file:
        # Iterate over each bin group
        for binID in hdf5_file.keys():
            IDString  = int(binID[4:])
            bin_group = hdf5_file[binID]
            
            # Initialize a dictionary to hold the data for this bin
            ZCDFDictSet[IDString] = {}
            
            # Each bin group contains 'r_###' subgroups
            for RID in bin_group.keys():
                r_group   = bin_group[RID]
                RIDString = int(RID[2:])
                
                # Initialize a dict for the data under this r-group
                data_dict = {}
                
                # Each r group has multiple datasets (originally keys in the data_dict)
                for dataset_key in r_group.keys():
                    # Read dataset into memory
                    data_dict[dataset_key] = r_group[dataset_key][...]  # "..." reads the entire dataset
                
                # Store this reconstructed dictionary
                ZCDFDictSet[IDString][RIDString] = data_dict
    return ZCDFDictSet

def load_Rdicts_from_hdf5(file_path):
    def quick_load(file_path):
        with h5py.File(file_path, 'r') as hdf5_file:
            group_names = sorted(hdf5_file.keys(), key=lambda x: int(x.split('_')[1]))
            for group_name in group_names:
                group = hdf5_file[group_name]
                data_dict = {dataset_name: group[dataset_name][:] for dataset_name in group}
                yield data_dict
    ModelRCache     = []
    for Dict in quick_load('./GalCache/BesanconRData.h5'):
        # Process each dictionary one at a time
        ModelRCache.append(Dict)    

    return ModelRCache
