"""
Bar plot of resolved DWD sources per binary-evolution code.

Adapted from Galaxy-scripts/lisa_dwd_counter.py: that one reads the raw catalog
CSVs and counts LISA-band candidates. This one reads our pipeline output
(${code}_output_cat.h5) and counts the *resolved foreground* — sources actually
subtracted by gwg.icloop.

Run from Foregrounds/ after main_loop.py has been run for the codes of interest:
    python compare_resolved.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import gwg

from helpers import load_and_prepare_config

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']


def count_resolved(outpath, code):
    fpath = os.path.join(outpath, f'{code}_output_cat.h5')
    if not os.path.exists(fpath):
        return np.nan
    return len(gwg.utils.load_h5(fpath, key='cat'))


if __name__ == "__main__":
    # local_run.sh sets this; set a sensible default for direct `python compare_resolved.py`
    os.environ.setdefault('EXPERIMENT_ROOT', './')
    config = load_and_prepare_config('config.yaml')
    outpath = os.path.join(config['outputpath'], config['datapath'])

    counts = {code: count_resolved(outpath, code) for code in CODES}
    print(f"{'code':8s}  resolved")
    for code, n in counts.items():
        print(f"{code:8s}  {n}")

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
