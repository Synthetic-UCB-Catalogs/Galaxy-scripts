"""
Per-cell confusion-foreground overlays: one plot per (code, variation).

For each (code, variation) cell in the analytic-fit coefficient table, overlay
    - the pipeline noise PSD  S[channel]   (the "noisy" curve, from the output h5)
    - the analytic fit        instr + R(f)*S_conf(theta)   (rebuilt from the CSV)
    - the TDI1 instrument reference
on one axis, and write it as
    <outroot>/<family>[/<subfamily>]/<variation>/<CODE> confusion foreground.png

The leading "<CODE> " (code name + space) is the key the ucb_plot_comparator app
parses as the Code field, with the rest becoming the Type. The folder tree mirrors
the catalog taxonomy so it lines up with the Code_comparison_plots tree.

Data sources:
  - noisy curve : {outputpath}/{datapath}/{code}_output_cat_snr{cutoff}.h5, key "S"
                  (columns A,E,T,f). These files are large (~1 GB each).
  - fit curve   : analytic_fits/fits/confusion_coeffs_karnesis.csv (theta per cell),
                  fully local. Rebuilt via fit_confusion.model_conf (no re-fitting).

Run in the pipeline env (needs lisatools for the instrument PSD and R(f)):
    python foreground_percell_plots.py --outputpath ./output --outroot ./Foreground_comparison_plots
Point --outputpath at wherever the output h5 live.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the fit model + lisatools helpers from the fitter (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytic_fits"))
from fit_confusion import instrument_aet, stochastic_response, model_conf  # noqa: E402

KARNESIS_PARAMS = ["A", "alpha", "f1", "f2", "fknee"]
ROBSON_PARAMS = ["A", "alpha", "beta", "kappa", "gamma", "fk"]
BAND = (5e-5, 2.5e-2)   # match compare_resolved's noise-curve load window
VIEW = (1e-4, 2e-2)     # match the existing overlay x-range


def datapath_for(row):
    """family[/subfamily]/variation, matching the catalog + Code_comparison_plots tree."""
    parts = [row["family"]]
    sub = row.get("subfamily")
    if isinstance(sub, str) and sub and sub.lower() != "nan":
        parts.append(sub)
    parts.append(row["variation"])
    return os.path.join(*parts)


def find_h5(outputpath, datapath, code, cutoff):
    """Locate {code}_output_cat*.h5 for a cell.

    The coeffs CSV taxonomy (family[/subfamily]/variation) sits UNDER an extra
    tree prefix in the output dir (monte_carlo_comparisons/...), so search
    recursively for the datapath leaf rather than assuming the prefix. Prefer the
    snr-suffixed file in the full (non-lightweight) tree.
    """
    pattern = os.path.join(outputpath, "**", datapath, f"{code}_output_cat*.h5")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    hits.sort(key=lambda p: ("lightweight" in p,                       # full tree first
                             f"snr{cutoff:g}" not in os.path.basename(p),  # snr-cutoff first
                             p))
    return hits[0]


def load_noisy(h5, channel):
    """Return (f, S[channel]) cropped to BAND and downsampled (~5000 pts; curve is smooth)."""
    S = pd.read_hdf(h5, key="S")
    f = np.asarray(S["f"], dtype=np.float64)
    Sa = np.abs(np.asarray(S[channel], dtype=np.float64))
    m = (f >= BAND[0]) & (f <= BAND[1])
    f, Sa = f[m], Sa[m]
    step = max(1, len(f) // 5000)
    return f[::step], Sa[::step]


def theta_from_row(row, model):
    names = KARNESIS_PARAMS if model == "karnesis" else ROBSON_PARAMS
    return [float(row[n]) for n in names]


def plot_cell(f, Sn, theta, model, channel, code, variation, out_png):
    instr = instrument_aet(f)[channel]
    fit = instr + stochastic_response(f) * model_conf(f, theta, model)

    with plt.rc_context({"font.size": 12, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.loglog(f, Sn, color="darkorange", lw=1.2, label=f"pipeline PSD ({channel})")
        ax.loglog(f, fit, color="red", lw=2.0, label="analytic fit")
        ax.loglog(f, instr, "k--", lw=1.5, alpha=0.7, label="instrument (TDI1)")
        ax.set_xlim(*VIEW)
        view = (f >= VIEW[0]) & (f <= VIEW[1])
        ymax = float(np.nanmax(Sn[view])) if view.any() else float(np.nanmax(Sn))
        ymin = max(float(np.nanmin(instr[view])) * 0.5, 1e-44) if view.any() else 1e-44
        ax.set_ylim(ymin, ymax * 3)
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel(r"PSD [1/Hz]")
        ax.text(0.98, 0.98, f"{code}\n{variation}", transform=ax.transAxes,
                ha="right", va="top", fontsize=13)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=":", linewidth=1.0)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=10)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--coeffs", default=os.path.join(here, "analytic_fits", "fits",
                                                     "confusion_coeffs_karnesis.csv"),
                    help="analytic-fit coefficient table")
    ap.add_argument("--outputpath", default=os.path.join(here, "output"),
                    help="root holding {datapath}/{code}_output_cat_snr*.h5")
    ap.add_argument("--outroot", default=os.path.join(here, "Foreground_comparison_plots"),
                    help="destination tree for the per-cell PNGs")
    ap.add_argument("--channel", default="A", choices=["A", "E", "T"])
    ap.add_argument("--snr-cutoff", type=float, default=7.0)
    ap.add_argument("--only", default=None, help="restrict to one code (for quick tests)")
    args = ap.parse_args()

    coeffs = pd.read_csv(args.coeffs)
    made, skipped = 0, 0
    for _, row in coeffs.iterrows():
        code = row["code"]
        if args.only and code != args.only:
            continue
        model = row["model"]
        dp = datapath_for(row)
        h5 = find_h5(args.outputpath, dp, code, args.snr_cutoff)
        if h5 is None:
            skipped += 1
            continue
        try:
            f, Sn = load_noisy(h5, args.channel)
            theta = theta_from_row(row, model)
            out_png = os.path.join(args.outroot, dp, f"{code} confusion foreground.png")
            plot_cell(f, Sn, theta, model, args.channel, code, row["variation"], out_png)
            print(f"saved {out_png}")
            made += 1
        except Exception as e:
            print(f"SKIP {code} @ {dp}: {e}")
            skipped += 1

    print(f"\n{made} plots written to {args.outroot} ; {skipped} cells skipped (no h5 / error)")


if __name__ == "__main__":
    main()
