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

    print(f"{'code':8s}  {'LISA (full)':>12s}  {'resolved':>10s}")
    for code in CODES:
        print(f"{code:8s}  {lisa_counts[code]:>12}  {counts[code]:>10}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(CODES)))
    values = [counts[c] for c in CODES]
    bars = ax.bar(CODES, values, color=colors, edgecolor='black')

    for bar, n in zip(bars, values):
        if not np.isnan(n):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{int(n)}', ha='center', va='bottom')

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
