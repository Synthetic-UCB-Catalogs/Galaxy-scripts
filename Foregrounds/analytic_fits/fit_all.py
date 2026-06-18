"""
Batch-run fit_confusion.py over every (leaf, code) and assemble the per-family fit
summaries + a coefficient table, mirroring compare_resolved.py's per-family grouping.

Decoupled from the fast plotter: this runs the (expensive) emcee fits and reads only their
JSON outputs. Layout under --out (median methoduse only; no mean/median split):

    confusion_coeffs_{model}.csv                  aggregate theta +/- errors, all (leaf, code)
    confusion_fits_{code}_{model}.png             ICV headline: per code, the 6 ICVs overlaid
    confusion_fits_{family}_{code}_{model}.png    MTV headline: per (subfamily, code), subvariations overlaid
    debug/<taxonomy>/<variation>/                 granular per-(code,leaf): the noisy pipeline curve +
                                                  analytic fit overlay, JSON, corner png
                                                  (full-disclosure / accompanying data repo)

The full post-burn-in MCMC chain + its config sidecar are written separately by fit_confusion
into a _chains/ subfolder of the catalog's own output/ leaf dir (symlinked to scratch, kept apart
from the .h5) -- NOT under --out -- so the large chains stay off the HOME quota and don't mix with
the pipeline outputs; they are there for later corner polishing.

Headline summaries overlay the fitted curves only (instr + R(f)*S_conf), gist_rainbow over
[fiducial baseline + variations] with reserved slots, so they pair 1:1 with the noise_curves_*
figures. The noisy-vs-fit comparison lives in the per-(code,leaf) debug overlay (fit_confusion's
own PNG: pipeline curve + median + coarse-grained points + fit).

The fits are emcee (expensive) -- run once; re-make the summaries + table cheaply with
--collect-only. Extra flags after the known ones forward to fit_confusion (e.g. --nsteps,
--n-per-decade, --kappa).

Examples (run from Foregrounds/):
    # all leaves, all codes (auto-discover every leaf under the tree):
    python analytic_fits/fit_all.py --root output/monte_carlo_comparisons --out analytic_fits/fits
    # explicit leaf dirs instead of --root:
    python analytic_fits/fit_all.py --out analytic_fits/fits \
        --dirs output/monte_carlo_comparisons/initial_condition_variations/fiducial
    # just rebuild summaries + table from fits already on disk:
    python analytic_fits/fit_all.py --root output/monte_carlo_comparisons --out analytic_fits/fits --collect-only
    # rebuild every per-leaf figure (overlay+corner) + JSON from the saved chains, no emcee
    # (e.g. after wiping analytic_fits/fits/ -- the chains + residual TDI survive on scratch):
    python analytic_fits/fit_all.py --root output/monte_carlo_comparisons --out analytic_fits/fits --replot
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

CODES = ["BPASS", "BSE", "ComBinE", "COMPAS", "COSMIC", "METISSE", "SeBa", "SEVN"]

# Variation grouping -- mirrors compare_resolved.py so the fit summaries pair 1:1 with the
# noise_curves_* figures (same series order and gist_rainbow colours per family).
ICV_VARIATIONS = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01', 'thermal_ecc', 'uniform_ecc']
MTV_FAMILIES = {
    "common_envelope": ['alpha_lambda_02', 'alpha_lambda_05', 'alpha_lambda_1',
                        'alpha_lambda_2', 'alpha_gamma_2'],
    "stability_of_mass_transfer": ['qcrit_claeys_14', 'qcrit_hurley_02',
                                   'qcrit_hurley_webbink', 'qcrit_zetas'],
    "stable_accretion_efficiency": ['accretion_0', 'accretion_05', 'accretion_1'],
}

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(HERE, "fit_confusion.py")
# Mirrors fit_confusion.EXIT_TOO_FEW_BINS: a deliberate below-confusion-floor skip, not a crash.
EXIT_BELOW_FLOOR = 3


def load_json(out_ic, code, model):
    jp = os.path.join(out_ic, f"{code}_confusion_fit_{model}.json")
    if not os.path.exists(jp):
        return None
    with open(jp) as fh:
        return json.load(fh)


def discover_leaf_dirs(root, snr):
    """Every dir under root holding a {code}_output_cat_snr{snr}.h5 (ICV + MTV leaves)."""
    pat = os.path.join(root, "**", f"*_output_cat_snr{snr:g}.h5")
    return sorted({os.path.dirname(p) for p in glob.glob(pat, recursive=True)})


def parse_taxonomy(d):
    """(family, subfamily, variation) from an output leaf dir path. subfamily is the MTV
    subfolder (common_envelope/...) or None for ICV leaves and the flat MTV fiducial."""
    parts = [p for p in d.rstrip("/").split(os.sep) if p]
    variation = parts[-1] if parts else "root"
    if "initial_condition_variations" in parts:
        return "initial_condition_variations", None, variation
    if "mass_transfer_variations" in parts:
        rest = parts[parts.index("mass_transfer_variations") + 1:]
        if len(rest) >= 2:
            return "mass_transfer_variations", rest[0], rest[-1]
        return "mass_transfer_variations", None, variation
    return "other", None, variation


def debug_dir(out, family, subfamily, variation):
    """Granular per-leaf output dir: {out}/debug/<taxonomy>/<variation>/."""
    parts = [out, "debug", family]
    if subfamily:
        parts.append(subfamily)
    parts.append(variation)
    return os.path.join(*parts)


def summary_fit_plot(out_png, series_paths, code, model, title):
    """Overlay the analytic fit curve (instr + R(f)*S_conf) for each variation in
    series_paths = [(variation, debug_dir), ...], gist_rainbow over the series with reserved
    slots + the instrument reference. Mirrors compare_resolved._noise_curves (fit curves only,
    no noisy tdi). Returns True if at least one curve was drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, HERE)
    import importlib
    fc = importlib.import_module("fit_confusion")

    f = np.logspace(-4, np.log10(2e-2), 800)
    instr = fc.instrument_aet(f)
    R = fc.stochastic_response(f)
    colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(series_paths)))
    ref, ymax, drawn = None, None, False
    with plt.rc_context({"font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12}):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, (var, ddir) in enumerate(series_paths):
            res = load_json(ddir, code, model)
            if res is None:
                ax.loglog([], [], color=colors[j], label=var)      # reserved slot (mirror bar plot)
                continue
            names = list(res["coefficients"].keys())
            theta = np.array([res["coefficients"][n]["median"] for n in names], dtype=float)
            ch = (res.get("channels") or ["A"])[0]
            ref = ref or ch
            tot = instr[ch] + R * fc.model_conf(f, theta, model)
            ax.loglog(f, tot, color=colors[j], lw=1.8, label=var)
            view = (f >= 1e-4) & (f <= 2e-2)
            ymax = max(ymax or float(instr[ch].min()), float(np.nanmax(tot[view])))
            drawn = True
        if not drawn:
            plt.close(fig)
            return False
        ref = ref or "A"
        ax.loglog(f, instr[ref], "k--", lw=1.2, alpha=0.6, label="instrument")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(float(instr[ref].min()) * 0.5, ymax * 2)
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel(r"PSD [1/Hz]")
        ax.text(0.98, 0.98, f"{title} ({ref})", transform=ax.transAxes,
                ha="right", va="top", fontsize=13)
        ax.legend(loc="upper left")
        ax.grid(True, which="both", linestyle=":", linewidth=1.0)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    print(f"  summary -> {out_png}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="auto-discover every leaf dir under this root holding a "
                         "{code}_output_cat_snr{snr}.h5 (ICV + MTV). Use this OR --dirs.")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="explicit output leaf dirs (ICV flat, MTV nested, and the flat MTV fiducial)")
    ap.add_argument("--codes", nargs="+", default=CODES)
    ap.add_argument("--snr", type=float, default=7)
    ap.add_argument("--model", default="karnesis", choices=("karnesis", "robson"))
    ap.add_argument("--out", default="fits", help="root output dir (median only; debug/ created under it)")
    ap.add_argument("--table", default=None, help="output CSV (default {out}/confusion_coeffs_{model}.csv)")
    ap.add_argument("--collect-only", action="store_true",
                    help="don't run fits; just rebuild summaries + table from JSONs on disk")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (leaf, code) fit whose JSON already exists -> resumable sweep "
                         "(resubmit after a preemption/wall-hit and it continues where it left off)")
    ap.add_argument("--replot", action="store_true",
                    help="don't fit; rebuild JSON + per-leaf overlay/corner from each stored chain "
                         "(forwards --replot to fit_confusion), then the per-family summaries + table. "
                         "Recovers everything after a figures/ wipe -- chains + TDI survive on scratch.")
    args, extra = ap.parse_known_args()

    if args.root:
        dirs = discover_leaf_dirs(args.root, args.snr)
        if not dirs:
            ap.error(f"no leaf dirs with *_output_cat_snr{args.snr:g}.h5 under {args.root}")
    elif args.dirs:
        dirs = args.dirs
    else:
        ap.error("provide --root (auto-discover) or --dirs (explicit leaf dirs)")

    # Resolve taxonomy per leaf and where its granular artifacts live.
    leaves = []
    for d in dirs:
        family, subfamily, variation = parse_taxonomy(d)
        ddir = debug_dir(args.out, family, subfamily, variation)
        leaves.append(dict(d=d, family=family, subfamily=subfamily, variation=variation, ddir=ddir))
    print(f"{len(leaves)} leaves, codes {args.codes}, model {args.model}, out {args.out}")

    # --- replot: rebuild JSON + per-leaf overlay/corner from each stored chain (no emcee) ---
    if args.replot:
        for lf in leaves:
            os.makedirs(lf["ddir"], exist_ok=True)
            for code in args.codes:
                chain = os.path.join(lf["d"], "_chains", f"{code}_confusion_fit_{args.model}_chain.npy")
                if not os.path.exists(chain):
                    continue
                cmd = [sys.executable, FIT, "--dir", lf["d"], "--code", code, "--snr", str(args.snr),
                       "--model", args.model, "--out", lf["ddir"], "--replot"] + extra
                rc = subprocess.run(cmd).returncode
                if rc == EXIT_BELOW_FLOOR:
                    print(f"[{lf['variation']}/{code}] below confusion floor (<10 bins); no replot")
                elif rc != 0:
                    print(f"[{lf['variation']}/{code}] replot failed; skipping")

    # --- run the (expensive) per-(leaf, code) fits into debug/<taxonomy>/ ---
    elif not args.collect_only:
        for lf in leaves:
            os.makedirs(lf["ddir"], exist_ok=True)
            for code in args.codes:
                if args.skip_existing and load_json(lf["ddir"], code, args.model) is not None:
                    print(f"[{lf['variation']}/{code}] JSON exists; skip (resume)")
                    continue
                h5 = os.path.join(lf["d"], f"{code}_output_cat_snr{args.snr:g}.h5")
                if not os.path.exists(h5):
                    print(f"[{lf['variation']}/{code}] no h5; skipping")
                    continue
                cmd = [sys.executable, FIT, "--dir", lf["d"], "--code", code, "--snr", str(args.snr),
                       "--model", args.model, "--out", lf["ddir"]] + extra
                rc = subprocess.run(cmd).returncode
                if rc == EXIT_BELOW_FLOOR:
                    print(f"[{lf['variation']}/{code}] below confusion floor (<10 bins); no fit")
                elif rc != 0:
                    print(f"[{lf['variation']}/{code}] fit_confusion failed; skipping")

    # --- aggregate coefficients table (read whatever JSONs exist) ---
    rows, coeff_names = [], None
    for lf in leaves:
        for code in args.codes:
            res = load_json(lf["ddir"], code, args.model)
            if res is None:
                continue
            coeff_names = list(res["coefficients"].keys())
            row = {"family": lf["family"], "subfamily": lf["subfamily"] or "",
                   "variation": lf["variation"], "code": code, "model": args.model,
                   "n_points": res.get("n_points"),
                   "fmin_hz": res["fit_band_hz"][0], "fmax_hz": res["fit_band_hz"][1]}
            for name, c in res["coefficients"].items():
                row[name] = c["median"]
                row[name + "_minus"] = c["minus"]
                row[name + "_plus"] = c["plus"]
            rows.append(row)

    os.makedirs(args.out, exist_ok=True)

    # --- headline per-family summaries (mirror the noise_curves grouping) ---
    # ICV: one figure per code, the 6 ICVs overlaid (fiducial baseline = gist_rainbow index 0).
    icv_ddir = {lf["variation"]: lf["ddir"] for lf in leaves
                if lf["family"] == "initial_condition_variations"}
    for code in args.codes:
        series = [(v, icv_ddir[v]) for v in ICV_VARIATIONS if v in icv_ddir]
        if series:
            summary_fit_plot(os.path.join(args.out, f"confusion_fits_{code}_{args.model}.png"),
                             series, code, args.model, f"ICV · {code}")

    # MTV: one figure per (subfamily, code), [MTV-fiducial baseline + the subfamily] overlaid.
    mtv_fid = next((lf["ddir"] for lf in leaves
                    if lf["family"] == "mass_transfer_variations" and lf["variation"] == "fiducial"), None)
    for family, subvars in MTV_FAMILIES.items():
        sub_ddir = {lf["variation"]: lf["ddir"] for lf in leaves
                    if lf["family"] == "mass_transfer_variations" and lf["subfamily"] == family}
        series = ([("fiducial", mtv_fid)] if mtv_fid else []) + \
                 [(v, sub_ddir[v]) for v in subvars if v in sub_ddir]
        if len(series) < 2:
            continue
        for code in args.codes:
            summary_fit_plot(os.path.join(args.out, f"confusion_fits_{family}_{code}_{args.model}.png"),
                             series, code, args.model, f"{family} · {code}")

    # --- write the table ---
    if not rows:
        print("No fits to tabulate.")
        return
    df = pd.DataFrame(rows)
    table = args.table or os.path.join(args.out, f"confusion_coeffs_{args.model}.csv")
    df.to_csv(table, index=False)
    print(f"\nwrote {len(df)} rows -> {table}")
    show = ["family", "subfamily", "variation", "code"] + (coeff_names or [])
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
