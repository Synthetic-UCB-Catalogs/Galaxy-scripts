import h5py 

class GalaxyModel:
    def __init__(self, name, modelParams, #CalculateGalacticComponentMassFractions=None, RhoFunction=None, 
                 recalculateNormConstants=False, recalculateCDFs=False):
        self.name = name
        self.modelParams = self.GetModelParameters()

        self.fnameComponentMassFractions = "ComponentMassFractions_{}.csv".format(name)
        self.fnameRZ_CDFData = "RZ_CDF_Data_{}.csv".format(name)
        if recalculateNormConstants or not os.path.isfile(self.fnameComponentMassFractions):
            self.CalculateGalacticComponentMassFractions()
        if recalculateCDFs or not os.path.isfile(self.fnameComponentMassFractions):
            self.CalculateRZ_CDFs()

    # Getters
    def getGalacticComponentMassFractions(self):
        return pd.read_csv(self.fnameComponentMassFractions).to_dict(orient='list')
        # TODO: change to hdf5

    def getRZ_CDFs(self):
        with h5py.File(self.fnameRZ_CDFData, 'r') as hdf5_file:
            group_names = sorted(hdf5_file.keys(), key=lambda x: int(x.split('_')[1]))
            for group_name in group_names:
                group = hdf5_file[group_name]
                data_dict = {dataset_name: group[dataset_name][:] for dataset_name in group}
                yield data_dict
        return load_RZdicts_from_hdf5()


    def GetModelParameters():
        pass
                
    # Setters
    def CalculateGalacticComponentMassFractions(self):
        pass

    def CalculateRZ_CDFs(self):
        pass

    #Routine to load data from a 2D-organised hdf5 file - TODO: is there an easier way to access hdf5 data?
    def load_RZdicts_from_hdf5(self):
        ZCDFDictSet = {}
        
        # Open the file for reading
        with h5py.File(self.fnameRZ_CDFData, 'r') as hdf5_file:
            # Iterate over each component group
            for componentID in hdf5_file.keys():
                IDString  = int(componentID[4:])
                component_group = hdf5_file[componentID]
                
                # Initialize a dictionary to hold the data for this component
                ZCDFDictSet[IDString] = {}
                
                # Each component group contains 'r_###' subgroups
                for RID in component_group.keys():
                    r_group   = component_group[RID]
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










