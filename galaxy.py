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
            self.T0_DWD_LISA = pop_create.get_possible_T0_LISA_sources(self.ModelParams, self.T0_dat_path, verbose=True)
        except Exception as e:
            raise IOError(f"Failed to load T0 data from {self.T0_dat_path}: {e}")


    def load_T0_data(self):
        """Loads T0 data from the specified path."""
        if self.T0_dat_path is None:
            raise ValueError("T0 data path is not specified.")
        
        try:
            # Assuming T0 data is in a format that can be read into a DataFrame
            T0_DWD = pop_create.get_T0_DWDs(self.T0_dat_path, verbose=True)
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
    #def calculate_CDFs(self):

    # Maybe also implement default CDF for project w/ Besancon model

    # Create Galaxy
    def create_galaxy(self, write_path=None, verbose=False, write_h5=False):
        """Creates a DataFrame containing present-day DWDs in the Galaxy."""
        if self.T0_DWD_LISA is None:
            raise ValueError("T0 DWD data is not loaded or does not contain 'DWD' column. Please load and filter the LISA-specific T0 data first.")
        
        # Calculate the number of DWDs in the Galaxy
        self.calculate_N_DWD_Gx()
        
        # Create the galaxy component DataFrame
        _ = pop_create.create_LISA_galaxy(self.T0_DWD_LISA, self.N_DWD_Gx, write_path, verbose=verbose, write_h5=write_h5)
        
        return None

    # Calculate SNRs from legwork
    #def calculate_SNRs(self, T0_DWD_LISA):
        # Implement SNR calculation based on the T0 DWD LISA sources

