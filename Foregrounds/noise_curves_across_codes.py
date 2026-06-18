"""
Overlay the final smoothed (noisy) total PSD S(f) ACROSS the eight BPS codes for a single
catalog leaf (one fixed initial-condition / mass-transfer variation), at the SNR>cutoff
recovery run.

This is the transpose of compare_resolved._noise_curves: that routine fixes a code and
overlays the variations; here we fix the variation (leaf) and overlay the codes. The curve
per code is the final smoothed total PSD stored by main_loop as key "S" (galactic confusion
+ instrument) in {code}_output_cat_snr{cutoff}.h5 -- so this must run where those outputs
live (Thorny). The figure format matches the LISA-Symposium poster (gist_rainbow over codes,
single-column legend, major gridlines only, ticks-in, TDI1 instrument reference).

Usage (from Foregrounds/):
    python noise_curves_across_codes.py \
        --datapath monte_carlo_comparisons/initial_condition_variations/fiducial/
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gwg

from helpers import load_and_prepare_config
import reference_snr

# Allow ad-hoc runs from Foregrounds/ without the sbatch / local_run.sh wrapper (which export
# EXPERIMENT_ROOT). config.yaml's outputpath is ${EXPERIMENT_ROOT}/output, so default it to "./"
# (resolved relative to config.yaml, i.e. the output/ symlink under Foregrounds/).
os.environ.setdefault("EXPERIMENT_ROOT", "./")

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']


def _load_S(h5, channel):
    """(f, |S_channel|) for the smoothed total PSD, masked to the confusion band + downsampled.
    Mirrors compare_resolved._noise_curves._load so the across-code curves match the per-code ones."""
    S = gwg.utils.load_h5(h5, key="S")
    fa = np.asarray(S["f"]); Sa = np.abs(np.asarray(S[channel]))
    m = (fa >= 5e-5) & (fa <= 2.5e-2)
    fa, Sa = fa[m], Sa[m]
    step = max(1, len(fa) // 5000)                     # the curve is smooth; downsample for speed
    return fa[::step], Sa[::step]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datapath", required=True,
                    help="catalog leaf under outputpath, e.g. "
                         "monte_carlo_comparisons/initial_condition_variations/fiducial/")
    ap.add_argument("--channel", default="A", choices=["A", "E", "T"])
    ap.add_argument("--snr-cutoff", type=float, default=7.0)
    ap.add_argument("--out", default=None,
                    help="output PNG (default figures/noise_curves_across_codes_{variation}.png)")
    args = ap.parse_args()

    config = load_and_prepare_config("config.yaml")
    tdi = 1 if not config.get("tdi2", False) else 2
    leaf = os.path.join(config["outputpath"], args.datapath)
    variation = os.path.basename(os.path.normpath(args.datapath))
    out = args.out or os.path.join("figures", f"noise_curves_across_codes_{variation}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    fref = np.logspace(-4, np.log10(2e-2), 2000)
    instr = np.asarray(reference_snr._instrument_psd_fn(tdi)(fref)[0])
    colors = {c: plt.cm.gist_rainbow(i / (len(CODES) - 1)) for i, c in enumerate(CODES)}

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ymax = float(instr.min())
    drawn = 0
    for c in CODES:
        h5 = os.path.join(leaf, f"{c}_output_cat_snr{args.snr_cutoff:g}.h5")
        if not os.path.exists(h5):
            print(f"[{c}] no {os.path.basename(h5)}; skipping")
            continue
        try:
            fa, Sa = _load_S(h5, args.channel)
        except Exception as e:
            print(f"[{c}] could not load key 'S' ({e}); skipping")
            continue
        ax.loglog(fa, Sa, color=colors[c], lw=3.0, label=c)
        iv = (fa >= 1e-4) & (fa <= 2e-2)
        if iv.any():
            ymax = max(ymax, float(np.nanmax(Sa[iv])))
        drawn += 1
    if not drawn:
        raise SystemExit(f"No {{code}}_output_cat_snr{args.snr_cutoff:g}.h5 with key 'S' under {leaf}")

    ax.loglog(fref, instr, "k--", lw=1.3, alpha=0.6, label="instrument")
    ax.set_xlim(1e-4, 2e-2)
    ax.set_ylim(max(float(instr.min()) * 0.5, 1e-44), ymax * 2)
    ax.set_xlabel(r"Frequency  $f$  [Hz]", fontsize=14)
    ax.set_ylabel(r"channel PSD  $S(f)$  [Hz$^{-1}$]", fontsize=13)
    ax.text(0.97, 0.97, f"{variation} ({args.channel})", transform=ax.transAxes,
            ha="right", va="top", fontsize=15)
    ax.grid(True, which="major", ls=":", alpha=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
    ax.legend(fontsize=11, ncol=1, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({drawn} codes, channel {args.channel}, SNR>{args.snr_cutoff:g})")


if __name__ == "__main__":
    main()
