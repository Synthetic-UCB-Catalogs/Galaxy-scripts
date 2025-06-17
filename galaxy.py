"""A Class that represents a galaxy."""

import numpy as np
import LISA_preselect

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
            T0_DWD_LISA = LISA_preselect.get_possible_T0_LISA_sources(self.ModelParams, self.T0_dat_path, verbose=True)
            return T0_DWD_LISA
        except Exception as e:
            raise IOError(f"Failed to load T0 data from {self.T0_dat_path}: {e}")

    # Implement a statistics method to calculate the number of DWDs in the Galaxy
    #def calculate_number_of_DWDs(self):

    # Implement calculation of CDFs
    #def calculate_CDFs(self):

    # Maybe also implement default CDF for project w/ Besancon model

    # Assign Galaxy positions
    #def assign_positions(self, n_points):

    # Calculate SNRs from legwork
    #def calculate_SNRs(self, T0_DWD_LISA):
        # Implement SNR calculation based on the T0 DWD LISA sources

