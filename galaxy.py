"""A Class that represents a Milky-Way-like galaxy."""

import numpy as np
import pop_create

__all__ = ['Galaxy']



class Galaxy:
    """A class representing a galaxy that has several different mass components.
    
    Attributes
    ----------
    ModelParams : dict
        A dictionary containing the parameters of the galaxy model, such as total mass, bulge mass, halo mass, etc.
        
        
    """
    
    def __init__(self, ModelParams, T0_dat_path=None):
        """Initializes the Galaxy with parameters from ModelParams."""
        self.ModelParams = ModelParams
        self.T0_dat_path = T0_dat_path
        # should write some tests here to ensure that the ModelParams contains all necessary keys

    def __repr__(self):
        return (f"Galaxy(ModelParams={self.ModelParams}")

    def load_possible_LISA_sources(self):
        """Loads T0 data from the specified path and filters it to find possible LISA sources"""
        if self.T0_dat_path is None:
            raise ValueError("T0 data path is not specified.")
        try:
            self.T0_DWD_LISA = pop_create.get_possible_T0_LISA_sources(self.ModelParams, self.T0_dat_path, verbose=False)
        except Exception as e:
            raise IOError(f"Failed to load T0 data from {self.T0_dat_path}: {e}")


    def load_T0_data(self):
        """Loads T0 data from the specified path."""
        if self.T0_dat_path is None:
            raise ValueError("T0 data path is not specified.")
        
        try:
            # Assuming T0 data is in a format that can be read into a DataFrame
            T0_DWD = pop_create.get_T0_DWDs(self.T0_dat_path, verbose=False)
            return T0_DWD
        except Exception as e:
            raise IOError(f"Failed to load T0 data from {self.T0_dat_path}: {e}")

    
    # Implement a statistics method to calculate the number of DWDs in the Galaxy
    def calculate_N_DWD_Gx(self):
        """Calculates the number of DWDs in the Galaxy based on the model parameters."""
        # check if the T0 DataFrame contains necessary columns for DWDs
        if self.T0_DWD_LISA is None:
            raise ValueError("T0 DWD data is not loaded or does not contain 'DWD' column. Please load and filter the LISA-specific T0 data first.")
        
        # based on the model parameters, and selected galaxy mass, calculate the number of DWDs in the Galaxy
        self.N_DWD_Gx = pop_create.get_N_Gx_sample(self.T0_DWD_LISA, self.ModelParams)

    # Implement calculation of CDFs
    def calculate_CDFs(self, verbose=False):
        import h5py
        # Maybe also implement default CDF for project w/ Besancon model
        #Get the R-CDFs
        if self.ModelParams['recalculate_cdfs']: 
            if verbose:
                print("Calculating the CDFs!")
            from cdf_scripts import PreCompute, GetZCDF
            
            #Recalculate the r CDFs first:
            ModelRCache     = []
            for i in range(10):
                ModelRCache.append(PreCompute(i,'Besancon'))
        
            # Create an HDF5 file
            with h5py.File('./GalCache/BesanconRData.h5', 'w') as hdf5_file:
                if verbose:
                    print('Caching R')
                for idx, data_dict in enumerate(ModelRCache):
                    # Create a group for each dictionary
                    group = hdf5_file.create_group(f'Rdict_{idx}')
                    # Store each list as a dataset within the group
                    for key, value in data_dict.items():
                        group.create_dataset(key, data=value, compression='gzip')
            if verbose:
                print('R Cache saved')
                        
            #Recalculate the z-CDFs:
            
            #Sampling points dimension 1
            iBinSampleSet = [i for i in range(10)]
        
            # Create another HDF5 file
            with h5py.File('./GalCache/BesanconRZData.h5', 'w') as hdf5_file:
                for iBin in iBinSampleSet:
                    if verbose:
                        print('Caching Bin ' + str(iBin+1))
                    # Create a group for each x value
                    x_group = hdf5_file.create_group(f'bin_{iBin+1}')
                    rSet    = ModelRCache[iBin]['MidRSet']
                    rIDs    = list(range(len(rSet)))
                    for rID in rIDs:
                        if (rID % 100) == 0:
                            print('rID '+ str(rID))
                        # Create a subgroup for each y value within the x group
                        y_group = x_group.create_group(f'r_{rID}')
                        # Compute the function output
                        data_dict = GetZCDF(rSet[rID], iBin,'Besancon')
                        # Store each list in the dictionary as a dataset
                        for key, value in data_dict.items():
                            y_group.create_dataset(key, data=value, compression='gzip')
        else:
            if verbose:
                print('Reclaculate CDFs is false')
            

        return None
    

    # Create Galaxy
    def create_galaxy(self, write_path=None, verbose=False, write_h5=False, midpoint=False):
        """Creates a DataFrame containing present-day DWDs in the Galaxy."""
        if self.T0_DWD_LISA is None:
            raise ValueError("T0 DWD data is not loaded or does not contain 'DWD' column. Please load and filter the LISA-specific T0 data first.")
        
        # append Code to the galaxy write path
        galaxy_write_path = write_path

        # Calculate the number of DWDs in the Galaxy
        self.calculate_N_DWD_Gx()

        # Compute the Bezanscon CDFs if needed
        if self.ModelParams['recalculate_cdfs']:
            _ = self.calculate_CDFs(verbose=verbose)
            if verbose:
                print('CDFs calculated!')
        
        # Create the galaxy component DataFrame
        _ = pop_create.create_LISA_galaxy(self.T0_DWD_LISA, self.N_DWD_Gx, self.ModelParams, write_path=galaxy_write_path, verbose=verbose, write_h5=write_h5, midpoint=midpoint)
        
        return None
    
    def create_downsampled_galaxy(self, write_path=None, verbose=False, write_h5=False, midpoint=False):
        """Creates a downsampled DataFrame containing present-day DWDs in the Galaxy."""
        if self.T0_DWD_LISA is None:
            raise ValueError("T0 DWD data is not loaded or does not contain 'DWD' column. Please load and filter the LISA-specific T0 data first.")
        
        # append Code to the galaxy write path
        galaxy_write_path = write_path

        # Calculate the number of DWDs in the Galaxy
        self.calculate_N_DWD_Gx()
        self.N_DWD_Gx = int(self.N_DWD_Gx / self.ModelParams['downsample_fac'])  # Downsample the number of DWDs by the specified factor
        if verbose:
            print(f"Downsampling the number of DWDs in the Galaxy to {self.N_DWD_Gx} by a factor of {self.ModelParams['downsample_fac']}.")
        # Compute the Bezanscon CDFs if needed
        if self.ModelParams['recalculate_cdfs']:
            _ = self.calculate_CDFs(verbose=verbose)
            if verbose:
                print('CDFs calculated!')
        
        # Create the downsampled galaxy component DataFrame
        _ = pop_create.create_LISA_galaxy(self.T0_DWD_LISA, self.N_DWD_Gx, self.ModelParams, write_path=galaxy_write_path, verbose=verbose, write_h5=write_h5, midpoint=midpoint)
        
        return None

