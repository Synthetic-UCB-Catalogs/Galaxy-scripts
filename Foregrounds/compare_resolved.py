"""
Bar plot of resolved DWD sources per binary-evolution code.

Adapted from Galaxy-scripts/lisa_dwd_counter.py. That script counts LISA-band
DWDs from raw catalogs (no confusion noise). This one counts the resolved
foreground from our pipeline output (with confusion noise), and prints the
raw LISA count alongside for reference.

The reference is read from the full catalog, not the lightweight 500K
downsample, so it stays comparable to standard published values.

Usage:
    python compare_resolved.py
"""
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import gwg

from helpers import load_and_prepare_config

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']


def count_resolved(outpath, code):
    fpath = os.path.join(outpath, f'{code}_output_cat.h5')
    if not os.path.exists(fpath):
        return np.nan
    return len(gwg.utils.load_h5(fpath, key='cat'))


def count_lisa_dwds(data_root, code):
    fpath = os.path.join(data_root, f'{code}_Galaxy_LISA_DWDs.csv')
    if not os.path.exists(fpath):
        return np.nan
    return len(pd.read_csv(fpath, usecols=[0]))


def read_run_config(outpath, code):
    fpath = os.path.join(outpath, f'{code}_run_config.yaml')
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # local_run.sh sets this; set a sensible default for direct `python compare_resolved.py`
    os.environ.setdefault('EXPERIMENT_ROOT', './')
    config = load_and_prepare_config('config.yaml')
    outpath = os.path.join(config['outputpath'], config['datapath'])
    # always read LISA count from full catalog, even when running on lightweight
    full_datapath = config['datapath'].replace(
        'monte_carlo_comparisons_lightweight_500K_DWDs', 'monte_carlo_comparisons'
    )
    lisa_data_root = os.path.join(config['basepath'], full_datapath)

    counts = {code: count_resolved(outpath, code) for code in CODES}
    lisa_counts = {code: count_lisa_dwds(lisa_data_root, code) for code in CODES}
    run_configs = {code: read_run_config(outpath, code) for code in CODES}

    print(f"{'code':8s}  {'Tobs':>4s}  {'dt':>4s}  {'LISA (full)':>12s}  {'resolved':>10s}  {'res/LISA':>9s}")
    for code in CODES:
        cfg = run_configs[code]
        tobs = str(cfg.get('duration', '?')) if cfg else '?'
        dt = str(cfg.get('dt', '?')) if cfg else '?'
        lisa_n = lisa_counts[code]
        res_n = counts[code]
        ratio_str = f'{res_n / lisa_n * 100:.1f}%' if (pd.notna(lisa_n) and pd.notna(res_n) and lisa_n) else 'nan'
        print(f"{code:8s}  {tobs:>4s}  {dt:>4s}  {lisa_n:>12}  {res_n:>10}  {ratio_str:>9s}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(CODES)))
    values = [counts[c] for c in CODES]
    ax.bar(CODES, values, color=colors, edgecolor='black')

    ax.set_xlabel('Binary evolution code')
    ax.set_ylabel('Resolved DWD sources')
    ax.set_title(f"Resolved foreground per code\n{config['datapath'].rstrip('/')}")
    ax.grid(True, linestyle=':', linewidth=1., axis='y')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    figpath = os.path.join('figures', 'compare_resolved.png')
    fig.savefig(figpath, dpi=150)
    print(f"\nFigure saved to {figpath}")

    ratios = [
        (counts[c] / lisa_counts[c] * 100)
        if (pd.notna(lisa_counts[c]) and pd.notna(counts[c]) and lisa_counts[c])
        else np.nan
        for c in CODES
    ]
    # Poisson sqrt(N_resolved) shot noise, scaled by N_LISA into a percent
    errors = [
        np.sqrt(counts[c]) / lisa_counts[c] * 100
        if (pd.notna(counts[c]) and pd.notna(lisa_counts[c])
            and counts[c] > 0 and lisa_counts[c] > 0)
        else np.nan
        for c in CODES
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(CODES, ratios, color=colors, edgecolor='black',
            yerr=errors, capsize=4, ecolor='black')

    ax2.set_xlabel('Binary evolution code')
    ax2.set_ylabel('Resolved / LISA-band DWDs [%]')
    ax2.set_title(f"Recovery fraction per code\n{config['datapath'].rstrip('/')}")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=':', linewidth=1., axis='y')
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig2.tight_layout()
    figpath2 = os.path.join('figures', 'compare_recovery.png')
    fig2.savefig(figpath2, dpi=150)
    print(f"Figure saved to {figpath2}")
