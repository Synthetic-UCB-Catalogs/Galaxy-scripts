"""
Generates verification plots for a gwg-compatible catalog using JAX.

This script is modular and allows for running different computational methods
to generate a spectrum from a source catalog. The methods can be selected
via command-line arguments.

Usage:
    python jax_benchmark/plot_catalog_spectrum.py --code <sim_code> [--run-fd] [--run-td]

Example (run both methods):
    python jax_benchmark/plot_catalog_spectrum.py --code COSMIC --run-fd --run-td
"""
import os
import sys
import argparse
import yaml
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- JAX Setup ---
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# --- Add parent directory to path to import helpers ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import Constants, load_and_prepare_config, apply_global_plot_settings

# ==============================================================================
# JAX COMPUTE KERNELS
# These are the low-level, JIT-compiled functions for each method.
# ==============================================================================

@jax.jit
def calculate_approx_fd_spectrum_batch(amplitudes, f0_T, fdot0_T2, K, F_T):
    """JIT kernel for the approximate Frequency-Domain method."""
    A, f0, fdot = amplitudes[:, None], f0_T[:, None], fdot0_T2[:, None]
    mag_sq = (0.5 * A / (K * jnp.sqrt(fdot + 1e-100)))**2
    mask = (F_T >= f0) & (F_T <= f0 + fdot)
    return jnp.sum(mag_sq * mask, axis=0)

@jax.jit
def calculate_td_waveform_batch(amplitudes, f0, fdot0, psi0, t_vector):
    """JIT kernel for the Time-Domain summation method."""
    A, f0, fdot0, psi0 = amplitudes[:, None], f0[:, None], fdot0[:, None], psi0[:, None]
    psi_t = 2 * jnp.pi * f0 * t_vector + jnp.pi * fdot0 * t_vector**2 + psi0
    return jnp.sum(A * jnp.cos(psi_t), axis=0)

# ==============================================================================
# SUBROUTINE IMPLEMENTATIONS
# Each function is a self-contained workflow for a single method.
# ==============================================================================

def run_approx_fd_method(config, plot_settings, cat_df, code, batch_size=1000, batching=True):
    """
    Calculates and plots the spectrum using the approximate FD summation method.
    """
    print("\n--- Running Subroutine: Approximate FD Method ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Setup ---
    duration_yr = float(config['duration'])
    T_obs = Constants.yr * duration_yr
    dt = float(config['dt'])
    f_nyquist = 1.0 / (2.0 * dt)
    df = 1.0 / T_obs
    F_Hz = np.arange(0, f_nyquist, df)

    # --- Batching Logic ---
    num_sources = len(cat_df)
    num_batches = int(np.ceil(num_sources / batch_size))
    if not batching:
        num_batches = 100
    print(f"Processing {num_sources} sources in {num_batches} batches of size {batch_size}.")

    K = duration_yr
    T_yr_val = Constants.yr
    F_T_jnp = jnp.asarray(F_Hz * T_yr_val)
    total_spectrum_np = np.zeros_like(F_Hz, dtype=np.float64)

    start_time = time.perf_counter()
    for i in tqdm(range(num_batches), desc="FD Method Batches"):
        batch_df = cat_df.iloc[i * batch_size:(i + 1) * batch_size]
        
        amplitudes = jnp.asarray(batch_df['Amplitude'].values)
        f0_T = jnp.asarray(batch_df['Frequency'].values * T_yr_val)
        fdot0_T2 = jnp.asarray(batch_df['FrequencyDerivative'].values * T_yr_val**2)
        
        batch_spectrum = calculate_approx_fd_spectrum_batch(amplitudes, f0_T, fdot0_T2, K, F_T_jnp)
        total_spectrum_np += np.asarray(batch_spectrum.block_until_ready())
        
    end_time = time.perf_counter()
    print(f"FD method finished in {end_time - start_time:.4f} seconds.")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(F_Hz, total_spectrum_np)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Approximate Power Spectrum $|h_F|^2$')
    ax.set_title(f'Approximate FD Spectrum of {code} Catalog')
    ax.set_xlim(1e-4, 1e-1)
    ax.grid(True, linestyle=':', linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)
    fig.tight_layout()
    
    spectrum_path = os.path.join(script_dir, f'{code}_approx_spectrum_jax.png')
    fig.savefig(spectrum_path, dpi=plot_settings.get('dpi', 300))
    print(f"FD spectrum plot saved to: {spectrum_path}")
    plt.close(fig)

def run_td_summation_method(config, plot_settings, cat_df, code, batch_size=100, batching=True):
    """
    Calculates and plots the spectrum using the TD summation + FFT method.
    """
    print("\n--- Running Subroutine: TD Summation + FFT Method ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Setup ---
    duration_yr = float(config['duration'])
    T_obs = Constants.yr * duration_yr
    dt = float(config['dt'])
    t_vector_np = np.arange(0, T_obs, dt, dtype=np.float64)
    N_timesteps = len(t_vector_np)
    print(f"Generated time vector with {N_timesteps:,} points.")
    
    # --- Batching Logic ---
    num_sources = len(cat_df)
    num_batches = int(np.ceil(num_sources / batch_size))
    if not batching:
        num_batches = 100
    print(f"Processing {num_sources} sources in {num_batches} batches of size {batch_size}.")

    h_total_td_jnp = jnp.zeros_like(t_vector_np)
    t_vector_jnp = jnp.asarray(t_vector_np)

    start_time = time.perf_counter()
    for i in tqdm(range(num_batches), desc="TD Method Batches"):
        batch_df = cat_df.iloc[i * batch_size:(i + 1) * batch_size]
        
        amplitudes = jnp.asarray(batch_df['Amplitude'].values)
        f0 = jnp.asarray(batch_df['Frequency'].values)
        fdot0 = jnp.asarray(batch_df['FrequencyDerivative'].values)
        psi0 = jnp.asarray(batch_df['InitialPhase'].values)
        
        h_batch = calculate_td_waveform_batch(amplitudes, f0, fdot0, psi0, t_vector_jnp)
        h_total_td_jnp += h_batch.block_until_ready()
        
    h_total_td_jnp.block_until_ready()
    end_time = time.perf_counter()
    print(f"TD summation finished in {end_time - start_time:.4f} seconds.")

    # --- Final FFT ---
    print("Performing Final Fourier Transform with JAX...")
    start_time = time.perf_counter()
    h_fd_jnp = jnp.fft.rfft(h_total_td_jnp) * dt
    power_spectrum_jnp = jnp.abs(h_fd_jnp)**2
    freq_fd_jnp = jnp.fft.rfftfreq(N_timesteps, d=dt)
    power_spectrum_jnp.block_until_ready()
    end_time = time.perf_counter()
    print(f"JAX FFT finished in {end_time - start_time:.4f} seconds.")

    freq_fd_hz = np.asarray(freq_fd_jnp)
    power_spectrum = np.asarray(power_spectrum_jnp)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq_fd_hz, power_spectrum)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Power Spectrum $|h_F|^2$ from TD Summation')
    ax.set_title(f'FD Spectrum of {code} Catalog (TD Method)')
    ax.set_xlim(1e-4, 1e-1)
    ax.grid(True, linestyle=':', linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)
    fig.tight_layout()
    
    spectrum_path = os.path.join(script_dir, f'{code}_spectrum_td_method.png')
    fig.savefig(spectrum_path, dpi=plot_settings.get('dpi', 300))
    print(f"Spectrum plot saved to: {spectrum_path}")
    plt.close(fig)

# ==============================================================================
# SETUP AND ORCHESTRATION
# ==============================================================================

def setup(code):
    """Handles all common setup tasks."""
    print("--- Setup: Loading Configuration and Data ---")

    # --- Auto-detect environment and set root path ---
    if os.getenv('SCRATCH'):
        os.environ['EXPERIMENT_ROOT'] = os.path.join(os.getenv('SCRATCH'), 'projects/ucb-catalogs/confusion_test')
    else:
        os.environ['EXPERIMENT_ROOT'] = './'
    print(f"INFO: Set EXPERIMENT_ROOT to: {os.environ['EXPERIMENT_ROOT']}")

    # --- Load Configs and Apply Plot Settings ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yaml')
    plot_config_path = os.path.join(script_dir, '..', 'plot_config.yaml')

    try:
        config = load_and_prepare_config(config_path)
        with open(plot_config_path, 'r') as f:
            plot_settings = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"FATAL: Could not find a required configuration file. Error: {e}")
        sys.exit(1)

    apply_global_plot_settings(plot_settings)

    # --- Load and Filter Catalog ---
    input_cat_path = os.path.join(config['inputpath'], config['datapath'], f'{code}_input_cat.h5')
    print(f"Loading catalog from: {input_cat_path}")
    try:
        cat_df = pd.read_hdf(input_cat_path, key='cat')
    except FileNotFoundError:
        print(f"FATAL: Catalog file not found at '{input_cat_path}'")
        sys.exit(1)
    print(f"Loaded {len(cat_df)} sources.")
    
    dt = float(config['dt'])
    f_nyquist = 1.0 / (2.0 * dt)
    initial_source_count = len(cat_df)
    cat_df = cat_df[cat_df['Frequency'] < f_nyquist].copy()
    print(f"Filtered {initial_source_count - len(cat_df)} sources above Nyquist. {len(cat_df)} sources remain.")

    return config, plot_settings, cat_df

def main():
    """Main function to parse arguments and orchestrate the selected subroutines."""
    parser = argparse.ArgumentParser(
        description="Generate verification plots for a GW catalog using different JAX-based methods."
    )
    parser.add_argument('--code', type=str, required=True, help='Name of the simulation code to use.')
    parser.add_argument('--run-fd', action='store_true', help='Run the approximate Frequency-Domain method.')
    parser.add_argument('--run-td', action='store_true', help='Run the Time-Domain summation + FFT method.')
    args = parser.parse_args()

    # Perform the common setup
    config, plot_settings, cat_df = setup(args.code)

    # --- Generate Common Plots (like the histogram) ---
    print("\n--- Generating Common Plots ---")
    
    # Define a helper for plot styles to keep code DRY
    def _apply_common_styles(fig, ax):
        ax.grid(True, linestyle=':', linewidth='1.')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)
        fig.tight_layout()

    # Plot 1: Frequency Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
<<<<<<< Updated upstream
    ax.hist(
        cat_df['Frequency'], bins=np.logspace(-5, -1, 100), density=True
        histtype='step', lw=2
    )
=======
    ax.hist(cat_df['Frequency'], bins=np.logspace(-5, -1, 100), density=True, histtype='step', lw=2)
>>>>>>> Stashed changes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Number of Sources')
    ax.set_title(f'Frequency Distribution of {args.code} Catalog ({len(cat_df)} sources)')
    
    _apply_common_styles(fig, ax)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hist_path = os.path.join(script_dir, f'{args.code}_frequency_histogram.png')
    fig.savefig(hist_path, dpi=plot_settings.get('dpi', 300))
    print(f"Histogram saved to: {hist_path}")
    plt.close(fig)

    # Check if any methods were selected
    if not args.run_fd and not args.run_td:
        print("WARNING: No methods selected to run. Use --run-fd or --run-td.")
        parser.print_help()
        sys.exit(0)

    # Run the selected subroutines
    if args.run_fd:
        run_approx_fd_method(config, plot_settings, cat_df, args.code)
    
    if args.run_td:
        run_td_summation_method(config, plot_settings, cat_df, args.code)
    
    print("\nSUCCESS: All selected subroutines have completed.")

if __name__ == "__main__":
    main()
