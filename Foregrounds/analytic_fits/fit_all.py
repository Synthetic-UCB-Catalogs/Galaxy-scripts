"""
Batch-run fit_confusion.py over codes (and ICVs) and collect the JSON coefficients
into one table.

For each ICV output dir given in --dirs, runs fit_confusion for each code that has a
{code}_output_cat_snr{snr}.h5, writing that ICV's outputs to {out}/{variation}/ (so
different ICVs don't overwrite each other), then aggregates every
{code}_confusion_fit_{model}.json into {out}/confusion_coeffs_{model}.csv.

Each fit_confusion already uses the CPU pool (auto-detected), so codes run sequentially
here (one fit using all allocated cores at a time). Extra flags after the known ones are
forwarded to fit_confusion (e.g. --nsteps, --nproc, --n-per-decade).

Examples (run from Foregrounds/):
    # one ICV, all 8 codes:
    python analytic_fits/fit_all.py --dirs output/<...>/fiducial --out analytic_fits/fits
    # several ICVs at once:
    python analytic_fits/fit_all.py --dirs output/<...>/{fiducial,thermal_ecc} --out analytic_fits/fits
    # just rebuild the table from fits already on disk:
    python analytic_fits/fit_all.py --dirs output/<...>/fiducial --out analytic_fits/fits --collect-only
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

CODES = ["BPASS", "BSE", "ComBinE", "COMPAS", "COSMIC", "METISSE", "SeBa", "SEVN"]
HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(HERE, "fit_confusion.py")


def load_json(out_ic, code, model):
    jp = os.path.join(out_ic, f"{code}_confusion_fit_{model}.json")
    if not os.path.exists(jp):
        return None
    with open(jp) as fh:
        return json.load(fh)


def make_cumulative_plots(out_ic, variation, model, n_band, codes):
    """Per-ICV summary overlay of just the fitted curves (instr + R(f)*S_conf) for all codes,
    one colour each -- no noisy tdi. Writes two figures:
      {variation}_confusion_fits_{model}.png       median fit per code
      {variation}_confusion_fits_band_{model}.png  median + n_band posterior-draw thin lines/code
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, HERE)
    import importlib
    fc = importlib.import_module("fit_confusion")   # model_conf, stochastic_response, instrument_aet

    fits = []
    for code in codes:
        res = load_json(out_ic, code, model)
        if res is None:
            continue
        names = list(res["coefficients"].keys())
        theta = np.array([res["coefficients"][n]["median"] for n in names], dtype=float)
        sp = os.path.join(out_ic, f"{code}_confusion_fit_{model}_samples.npy")
        samples = np.load(sp) if os.path.exists(sp) else None
        ch = (res.get("channels") or ["A"])[0]
        fits.append((code, theta, samples, ch))
    if not fits:
        return

    f = np.logspace(-4, -2, 800)
    instr = fc.instrument_aet(f)
    R = fc.stochastic_response(f)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ref = fits[0][3]
    band_view = (f >= 2e-4) & (f <= 5e-3)

    def total(theta, ch):
        return instr[ch] + R * fc.model_conf(f, theta, model)

    for kind in ("fits", "band"):
        with plt.rc_context({"font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12,
                             "ytick.labelsize": 12, "legend.fontsize": 10}):
            fig, ax = plt.subplots(figsize=(9, 6))
            ymax = float(instr[ref].min())
            for i, (code, theta, samples, ch) in enumerate(fits):
                c = colors[i % 10]
                if kind == "band" and samples is not None and len(samples):
                    sel = np.random.default_rng(0).choice(len(samples), min(n_band, len(samples)), replace=False)
                    for th in samples[sel]:
                        ax.loglog(f, total(th, ch), color=c, lw=0.4, alpha=0.12)
                tot = total(theta, ch)
                ax.loglog(f, tot, color=c, lw=2, label=code)
                ymax = max(ymax, float(np.nanmax(tot[band_view])))
            ax.loglog(f, instr[ref], "k--", lw=1.0, alpha=0.6, label="instrument")
            ax.set_xlim(1e-4, 1e-2)
            ax.set_ylim(float(instr[ref].min()) * 0.5, ymax * 3)
            ax.set_xlabel(r"Frequency [Hz]")
            ax.set_ylabel(r"PSD [1/Hz]")
            ax.text(0.98, 0.98, f"{variation} ({ref})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=13)
            ax.legend(loc="upper left", ncol=2)
            ax.grid(True, which="both", linestyle=":", linewidth=1.0)
            ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=8)
            fig.tight_layout()
            suffix = "fits" if kind == "fits" else "fits_band"
            out_png = os.path.join(out_ic, f"{variation}_confusion_{suffix}_{model}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[{variation}] cumulative {kind} plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="one or more ICV output dirs, each holding {code}_output_cat_snr{snr}.h5")
    ap.add_argument("--codes", nargs="+", default=CODES)
    ap.add_argument("--snr", type=float, default=7)
    ap.add_argument("--model", default="karnesis", choices=("karnesis", "robson"))
    ap.add_argument("--out", default="fits", help="root output dir (per-ICV subdirs created under it)")
    ap.add_argument("--table", default=None, help="output CSV (default {out}/confusion_coeffs_{model}.csv)")
    ap.add_argument("--collect-only", action="store_true",
                    help="don't run fits; just aggregate JSONs already on disk")
    ap.add_argument("--n-band", type=int, default=50,
                    help="posterior draws per code in the cumulative band plot")
    args, extra = ap.parse_known_args()

    rows = []
    coeff_names = None
    for d in args.dirs:
        variation = os.path.basename(d.rstrip("/")) or "root"
        out_ic = os.path.join(args.out, variation)
        os.makedirs(out_ic, exist_ok=True)
        for code in args.codes:
            h5 = os.path.join(d, f"{code}_output_cat_snr{args.snr:g}.h5")
            if not args.collect_only:
                if not os.path.exists(h5):
                    print(f"[{variation}/{code}] no h5 ({h5}); skipping")
                    continue
                cmd = [sys.executable, FIT, "--dir", d, "--code", code, "--snr", str(args.snr),
                       "--model", args.model, "--out", out_ic] + extra
                if subprocess.run(cmd).returncode != 0:
                    print(f"[{variation}/{code}] fit_confusion failed; skipping")
                    continue
            res = load_json(out_ic, code, args.model)
            if res is None:
                if args.collect_only:
                    print(f"[{variation}/{code}] no JSON; skipping")
                continue
            coeff_names = list(res["coefficients"].keys())
            row = {"variation": variation, "code": code, "model": args.model,
                   "n_points": res.get("n_points"),
                   "fmin_hz": res["fit_band_hz"][0], "fmax_hz": res["fit_band_hz"][1]}
            for name, c in res["coefficients"].items():
                row[name] = c["median"]
                row[name + "_minus"] = c["minus"]
                row[name + "_plus"] = c["plus"]
            rows.append(row)
        make_cumulative_plots(out_ic, variation, args.model, args.n_band, args.codes)

    if not rows:
        print("No fits to tabulate.")
        return
    df = pd.DataFrame(rows)
    table = args.table or os.path.join(args.out, f"confusion_coeffs_{args.model}.csv")
    df.to_csv(table, index=False)
    print(f"\nwrote {len(df)} rows -> {table}")
    show = ["variation", "code"] + (coeff_names or [])
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
