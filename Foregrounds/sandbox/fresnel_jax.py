"""
A numerically robust JAX implementation of a complex Fresnel integral expression.

This script provides a JAX-jitted function to calculate the expression:
F(xi) = [Z(y+xi) + Z(y-xi)]/xi * exp(-0.5*i*pi*(y+xi)**2)
where y = nu*K/xi - xi.

This version uses the native `jax.scipy.special.fresnel` for a pure,
end-to-end JAX implementation.

Usage:
    python robust_fresnel_jax.py --config <path_to_config.yml>
"""
import os
import sys
import shutil
import argparse
import yaml

# --- JAX and Plotting Imports ---
import jax
import jax.numpy as jnp
from jax.scipy.special import fresnel # Use the native JAX implementation
jax.config.update("jax_enable_x64", True)

# --- Standard Library Imports ---
import numpy as np
import matplotlib.pyplot as plt

# --- Add parent directory to path to import helpers ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import load_and_prepare_config, apply_global_plot_settings


# --- Core JAX Implementation ---

def Z_jax(y):
    """
    Computes the complex Fresnel integral Z(y) = C(y) + iS(y) using
    the native JAX implementation.
    """
    s, c = fresnel(y)
    return c + 1j * s

@jax.jit
def robust_fresnel_expr_jax(xi, nu, K, threshold=15.0):
    """
    JAX-jitted version of the numerically robust expression calculation.

    Args:
        xi (JAX Array): The small parameter, must be positive.
        nu (float): A positive real number.
        K (float): A positive real number.
        threshold (float): Argument value to switch from asymptotic to direct.

    Returns:
        JAX Array: The complex result of the expression.
    """
    # --- Asymptotic (small xi) Regime Calculation ---
    term = jnp.pi * nu * K
    asymptotic_result = 2 * jnp.exp(-1j * term) * jnp.sinc(term / jnp.pi)

    # --- Direct (large xi) Regime Calculation ---
    y = (nu * K) / xi - xi
    arg1 = y + xi
    arg2 = y - xi
    exp_term = jnp.exp(-0.5j * jnp.pi * arg1**2)
    
    # Use the pure JAX Fresnel function. No callback needed.
    numerator = Z_jax(arg1) + Z_jax(arg2)
    direct_result = (numerator / xi) * exp_term

    # --- Hybrid Logic ---
    condition = ((nu * K) / xi) > threshold
    return jnp.where(condition, asymptotic_result, direct_result)


# --- Main Execution and Plotting Block ---
def main():
    parser = argparse.ArgumentParser(description="Robust JAX computation of a Fresnel expression.")
    args = parser.parse_args()

    # --- 1. Setup ---
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

    nu = 0.5
    K = 1.0

    # --- 2. JAX Computation ---
    print("Performing JAX computation...")
    xi_values_jax = jnp.logspace(-4, 2, 1000)
    robust_values_jax = robust_fresnel_expr_jax(xi_values_jax, nu, K)

    # --- 3. Plotting ---
    print("Generating plot...")
    xi_values_np = np.array(xi_values_jax)
    robust_values_np = np.array(robust_values_jax)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(xi_values_np, np.abs(np.real(robust_values_np)), label='Real Part (Abs)')
    ax.loglog(xi_values_np, np.abs(np.imag(robust_values_np)), label='Imaginary Part (Abs)')
    ax.loglog(xi_values_np, np.abs(robust_values_np), label='Absolute Value')

    ax.set_xlabel('Parameter $\\xi$')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Robust JAX Implementation ($\\nu={nu}, K={K}$)')
    ax.set_xlim(1e-4, 1e2)
    ax.grid(True, linestyle=':', linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)
    ax.legend()
    fig.tight_layout()
    
    spectrum_path = os.path.join(script_dir, f'robust_fresnel_jax_demo.png')
    fig.savefig(spectrum_path, dpi=plot_settings.get('dpi', 300))
    print(f"Plot saved to: {spectrum_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()