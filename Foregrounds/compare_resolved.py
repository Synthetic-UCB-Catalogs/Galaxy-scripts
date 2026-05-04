"""
Bar plot of resolved DWD sources per binary-evolution code.

Adapted from Galaxy-scripts/lisa_dwd_counter.py. That script counts LISA-band
DWDs from raw catalogs (no confusion noise). This one counts the resolved
foreground from our pipeline output (with confusion noise), and prints the
raw LISA count alongside for reference.

Color is per variation (fiducial, m2_min_05, ...) — same convention as
`lisa_dwd_count_plotter` upstream. Bars for variations without output yet
are plotted as NaN so the layout reserves room for them.

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
VARIATIONS = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01',
              'thermal_ecc', 'uniform_ecc']


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


def datapath_for_variation(datapath, variation):
    parts = datapath.rstrip('/').split('/')
    parts[-1] = variation
    return '/'.join(parts) + '/'


if __name__ == "__main__":
    # local_run.sh sets this; set a sensible default for direct `python compare_resolved.py`
    os.environ.setdefault('EXPERIMENT_ROOT', './')
    config = load_and_prepare_config('config.yaml')

    # Read counts for every (code, variation) pair; missing files → NaN.
    counts = {}
    lisa_counts = {}
    run_configs = {}
    for var in VARIATIONS:
        var_datapath = datapath_for_variation(config['datapath'], var)
        # always read LISA count from full catalog, even when running on lightweight
        var_lisa_datapath = datapath_for_variation(
            config['datapath'].replace('_lightweight_500K_DWDs', ''), var
        )
        outpath_var = os.path.join(config['outputpath'], var_datapath)
        lisa_root_var = os.path.join(config['basepath'], var_lisa_datapath)
        for code in CODES:
            counts[(code, var)] = count_resolved(outpath_var, code)
            lisa_counts[(code, var)] = count_lisa_dwds(lisa_root_var, code)
        run_configs[var] = {code: read_run_config(outpath_var, code) for code in CODES}

    # Print one table per variation that has any resolved output.
    available_vars = [v for v in VARIATIONS
                      if any(pd.notna(counts[(c, v)]) for c in CODES)]
    for var in available_vars:
        print(f"\n=== {var} ===")
        print(f"{'code':8s}  {'Tobs':>4s}  {'dt':>4s}  {'LISA (full)':>12s}  {'resolved':>10s}  {'res/LISA':>9s}")
        for code in CODES:
            cfg = run_configs[var].get(code)
            tobs = str(cfg.get('duration', '?')) if cfg else '?'
            dt = str(cfg.get('dt', '?')) if cfg else '?'
            lisa_n = lisa_counts[(code, var)]
            res_n = counts[(code, var)]
            ratio_str = f'{res_n / lisa_n * 100:.1f}%' if (pd.notna(lisa_n) and pd.notna(res_n) and lisa_n) else 'nan'
            print(f"{code:8s}  {tobs:>4s}  {dt:>4s}  {lisa_n:>12}  {res_n:>10}  {ratio_str:>9s}")

    # Title strips the trailing variation so it labels the dataset, not one ICV.
    parts = config['datapath'].rstrip('/').split('/')
    dataset_label = '/'.join(parts[:-1]) if len(parts) > 1 else config['datapath']

    width = 0.7 / len(VARIATIONS)
    colors = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(VARIATIONS)))
    xtick_pos = np.linspace(
        (len(VARIATIONS) / 2 - 0.5) * width,
        len(CODES) - 1 + (len(VARIATIONS) / 2 - 0.5) * width,
        len(CODES),
    )

    # ----- Resolved counts -----
    fig, ax = plt.subplots(figsize=(10, 5))
    labeled = set()
    for i, code in enumerate(CODES):
        for j, var in enumerate(VARIATIONS):
            val = counts[(code, var)]
            label = var if (var not in labeled and pd.notna(val)) else None
            ax.bar(i + j * width, val, width, color=colors[j],
                   edgecolor='black', label=label)
            if label:
                labeled.add(var)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(CODES)
    ax.set_xlabel('Binary evolution code')
    ax.set_ylabel('Resolved DWD sources')
    ax.set_title(f"Resolved foreground per code\n{dataset_label}")
    ax.legend()
    ax.grid(True, linestyle=':', linewidth=1., axis='y')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    figpath = os.path.join('figures', 'compare_resolved.png')
    fig.savefig(figpath, dpi=150)
    print(f"\nFigure saved to {figpath}")

    # ----- Recovery fraction (resolved / LISA, percent) with sqrt(N) error bars -----
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    labeled2 = set()
    for i, code in enumerate(CODES):
        for j, var in enumerate(VARIATIONS):
            res_n = counts[(code, var)]
            lisa_n = lisa_counts[(code, var)]
            if pd.notna(res_n) and pd.notna(lisa_n) and lisa_n > 0 and res_n > 0:
                ratio = res_n / lisa_n * 100
                err = np.sqrt(res_n) / lisa_n * 100
            else:
                ratio = np.nan
                err = 0.0  # NaN here triggers a matplotlib warning; bar is NaN anyway
            label = var if (var not in labeled2 and pd.notna(ratio)) else None
            ax2.bar(i + j * width, ratio, width, color=colors[j], edgecolor='black',
                    yerr=err, capsize=4, ecolor='black', label=label)
            if label:
                labeled2.add(var)
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(CODES)
    ax2.set_xlabel('Binary evolution code')
    ax2.set_ylabel('Resolved / LISA-band DWDs [%]')
    ax2.set_title(f"Recovery fraction per code\n{dataset_label}")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, linestyle=':', linewidth=1., axis='y')
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params('both', length=3, width=0.5, which='both', direction='in', pad=10)

    fig2.tight_layout()
    figpath2 = os.path.join('figures', 'compare_recovery.png')
    fig2.savefig(figpath2, dpi=150)
    print(f"Figure saved to {figpath2}")
