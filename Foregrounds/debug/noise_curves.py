"""Compare instrumental noise curves from `lisatools` and `fomweb`.

This script generates and plots the power spectral density (PSD) of the
observational channels (A, E, T) as calculated by the `lisatools` package
(using the SciRDv1 model) and the `fomweb` package.

It produces two output plots:
1. `noise_curve_comparison.png`: A direct comparison of the PSDs.
2. `noise_curve_relative_difference.png`: The relative difference between the two models.

Usage:
    python debug/noise_curves.py
"""

import numpy as np
import matplotlib.pyplot as plt

from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models
from fomweb.psd import Sh_A, Sh_E, Sh_T

# --- Configuration ---
F_MIN = 1e-5
F_MAX = 1.0
N_POINTS = 10000
# Define a common frequency vector for a fair comparison
f_vec = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_POINTS)

# --- 1. Generate Noise Curve from `lisatools` ---
print("Generating noise curve using `lisatools`...")
# Use the same SciRDv1 model as in the main script
sens_mat_lisatools = AET1SensitivityMatrix(f_vec, model=lisa_models.scirdv1, return_type='PSD')
noise_lisatools = {
    'f': f_vec,
    'A': sens_mat_lisatools[0],
    'E': sens_mat_lisatools[1],
    'T': sens_mat_lisatools[2]
}
print("... `lisatools` curve generated.")

# --- 2. Generate Noise Curve from `fomweb` ---
print("Generating noise curve using `fomweb`...")
# fomweb functions directly return the PSDs
noise_fomweb = {
    'f': f_vec,
    'A': Sh_A(f_vec),
    'E': Sh_E(f_vec),
    'T': Sh_T(f_vec)
}
print("... `fomweb` curve generated.")

# --- 3. Plot the Comparison ---
print("Plotting comparison...")
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
fig.suptitle('LISA Instrumental Noise PSD Comparison: `lisatools` vs. `fomweb`', fontsize=16)

channels = ["A", "E", "T"]
colors = {'lisatools': 'blue', 'fomweb': 'red'}

for i, chan in enumerate(channels):
    ax = axes[i]
    
    # Plot lisatools version
    ax.loglog(noise_lisatools['f'], noise_lisatools[chan], 
              label='`lisatools` (scirdv1)', color=colors['lisatools'], lw=3, alpha=0.8)
    
    # Plot fomweb version
    ax.loglog(noise_fomweb['f'], noise_fomweb[chan], 
              label='`fomweb`', color=colors['fomweb'], lw=2, linestyle='--')
    
    ax.set_title(f'Channel {chan}')
    ax.set_ylabel(r'PSD [$1/\mathrm{Hz}$]')
    ax.grid(True, which="both", ls=":")
    ax.legend()

axes[-1].set_xlabel(r'Frequency [Hz]')
fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

output_filename = 'noise_curve_comparison.png'
fig.savefig(output_filename)
print(f"Comparison plot saved to {output_filename}")

# --- 4. Plot the Relative Difference ---
print("Plotting relative difference...")
fig_rel, ax_rel = plt.subplots(figsize=(10, 6))
for chan in channels:
    # Calculate relative difference: (lisatools - fomweb) / fomweb
    relative_diff = (noise_lisatools[chan] - noise_fomweb[chan]) / noise_fomweb[chan]
    ax_rel.semilogx(f_vec, relative_diff * 100, label=f'Channel {chan}')

ax_rel.set_title(r'Relative Difference: (`lisatools` - `fomweb`) / `fomweb`')
ax_rel.set_xlabel('Frequency [Hz]')
ax_rel.set_ylabel('Relative Difference [%]')
ax_rel.grid(True, which="both", ls=":")
ax_rel.legend()

output_filename_rel = 'noise_curve_relative_difference.png'
fig_rel.tight_layout()
fig_rel.savefig(output_filename_rel)
print(f"Relative difference plot saved to {output_filename_rel}")
plt.close(fig)
plt.close(fig_rel)