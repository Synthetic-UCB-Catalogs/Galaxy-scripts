import pandas as pd
import matplotlib.pyplot as plt

import os
import hashlib

class Constants:
    G = 6.67e-11  # SI
    c = 3e+8    # m/s
    Msun = 2e+30  # kg
    AU = 1.5e+11 #  m

    # geometric units G=c=1 [everything is in seconds]
    Msun *= G/c**3   # s
    AU /= c  # s
    pc = 206265*AU   # s

    hr = 3600. # s
    yr = 365*24*hr  # s

def explore_csv(code, config):
    """
    Prints the structure and head of a CSV file using pandas.
    """

    base_path = config['basepath']
    data_path = config['datapath']
    filepath = os.path.join(
        base_path,
        data_path,
        f'{code}_Galaxy_AllDWDs.csv'
    )
    
    try:
        print(f"Exploring: {filepath}")
        df = pd.read_csv(filepath)
        print("\n--- File Info ---")
        df.info()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return df

def get_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def apply_global_plot_settings(plot_settings):
    """Applies matplotlib rc settings from a plot_settings dictionary."""
    MEDIUM_SIZE = plot_settings.get('medium_size', 14)
    BIGGER_SIZE = plot_settings.get('bigger_size', 18)

    plt.rcdefaults()
    try:
        plt.rc('text', usetex=True)
    except RuntimeError:
        print("Warning: LaTeX not found. Plots will be generated without it.")
        plt.rc('text', usetex=False)

    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
