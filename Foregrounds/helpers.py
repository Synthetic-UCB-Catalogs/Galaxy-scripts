import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import yaml
import hashlib
from pathlib import Path

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


def load_and_prepare_config(config_path):
    """
    Loads a YAML config, expands environment variables, and resolves relative paths.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The fully processed configuration dictionary.
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # This recursive function walks through the config to process all strings
    def process_paths_recursive(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = process_paths_recursive(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = process_paths_recursive(item)
        elif isinstance(data, str):
            # Step 1: Expand any environment variables (e.g., ${EXPERIMENT_ROOT})
            expanded_str = os.path.expandvars(data)
            
            # Step 2: If the path is still relative (starts with './'),
            # make it absolute with respect to the project root.
            if expanded_str.startswith('./'):
                return str(project_root / expanded_str)
            return expanded_str
            
        return data

    return process_paths_recursive(config)

    
def format_bytes(size_bytes):
    """Converts a size in bytes to a human-readable string."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

  
