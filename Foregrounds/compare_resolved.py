"""
Recovery-fraction analysis: resolved (gwg.icloop) / reference,
following Wouter's lisa_dwd_counter.py (`lisa_dwd_count_plotter`) convention.

Per-code grouped bars colored by ICV (initial-condition variation). The SNR-
threshold uncertainty (sweep of snr_cutoff) is shown ON TOP as error bars: each
bar is the nominal SNR>7 value; the error bar spans the cut5..cut9 range. ICVs
without output get reserved (empty) slots, matching the upstream plotter.

Figures (recovery plots share the resolved-counts legend, so only the counts
figure carries it — the recovery figures sit beside it in the draft):
    figures/compare_resolved.png                    resolved DWD counts per code (bars by ICV)
    figures/compare_recovery.png                    recovery % vs no-conf gbgpu reference
    figures/compare_recovery_legwork.png            recovery % vs no-conf LEGWORK reference
    figures/compare_recovery_legwork_robson19.png   recovery % vs LEGWORK robson19 reference

Per (code, ICV, cutoff): resolved from {code}_output_cat_snr{X}.h5; the references
(gbgpu per-source no-conf, LEGWORK no-conf, LEGWORK robson19) from reference_snr.py,
cached per code in each ICV's output dir and tagged by (tobs,dt,tdi). Run from
Foregrounds/. First run per ICV computes the references (expensive); later runs
read the cache.

Usage:
    python compare_resolved.py
"""
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gwg

from helpers import load_and_prepare_config
import reference_snr

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']
VARIATIONS = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01',
              'thermal_ecc', 'uniform_ecc']
MONEY_CUTOFF = 7.0

# Recovery references: (method key in the reference cache, y-axis label, output file, ylim).
# gbgpu/LEGWORK no-conf are the resolvability ceilings (<=100%); the LEGWORK robson19
# reference is suppressed by the galactic foreground, so its recovery can exceed 100%
# (autoscaled, ylim=None) — that overshoot is the point of the CSV cross-check.
RECOVERY_REFS = [
    ("gbgpu",            "Resolved / no-conf (gbgpu) [%]",    "compare_recovery.png",                  (0, 100)),
    ("legwork",          "Resolved / no-conf (LEGWORK) [%]",  "compare_recovery_legwork.png",          (0, 100)),
    ("legwork_robson19", "Resolved / robson19 (LEGWORK) [%]", "compare_recovery_legwork_robson19.png", None),
]


def datapath_for_variation(datapath, variation):
    parts = datapath.rstrip('/').split('/')
    parts[-1] = variation
    return '/'.join(parts) + '/'


def discover_runs(outpath, code):
    """[(cutoff, cat_path)] for each snr-cutoff run of `code` in `outpath`."""
    runs = []
    for cat_path in sorted(glob.glob(os.path.join(outpath, f"{code}_output_cat_snr*.h5"))):
        m = re.search(r"_output_cat_snr([0-9.]+)\.h5$", os.path.basename(cat_path))
        if m:
            runs.append((float(m.group(1)), cat_path))
    return runs


def resolved_count(cat_path):
    return len(gwg.utils.load_h5(cat_path, key="cat"))


def load_reference(code, config, outpath, inpath, datapath, thresholds, use_gpu):
    """Per-code reference-count DataFrame, cached. None if no input_cat.

    Methods: gbgpu (no-conf), legwork (no-conf), legwork_robson19 (galactic FG on).
    The robson19 reference is the CSV cross-check (TODO #2); it shares the no-conf
    LEGWORK source params and only differs in the confusion model.

    Caching is INCREMENTAL per method: a valid cache that only lacks a (cheap
    LEGWORK) method has just that method computed and appended — the expensive
    gbgpu per-source SNRs are never recomputed to add a reference.
    """
    cache_path = os.path.join(outpath, f"{code}_reference_counts.csv")
    input_cat = os.path.join(inpath, f"{code}_input_cat.h5")
    alldwds = os.path.join(config["basepath"], datapath, f"{code}_Galaxy_AllDWDs.csv")
    tobs = float(config["duration"]); dt = float(config["dt"])
    tdi = 1 if not config.get("tdi2", False) else 2
    need = {float(t) for t in thresholds}
    WANTED = ("gbgpu", "legwork", "legwork_robson19")
    if not os.path.exists(input_cat):
        print(f"  [{code}] input_cat missing; skipping reference")
        return None

    # Reuse any valid cached rows; recompute ONLY the methods actually missing.
    # gbgpu (per-source SNR) is the expensive one — never recompute it just to add
    # a new (cheap LEGWORK) reference method to an otherwise-valid cache.
    kept = pd.DataFrame()
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(input_cat):
        c = pd.read_csv(cache_path)
        tag_ok = len(c) and bool(((c.tobs_yr == tobs) & (c.dt_s == dt) & (c.tdi == tdi)).all())
        thr_ok = need.issubset(set(c.snr_threshold.astype(float)))
        if tag_ok and thr_ok:
            kept = c
    have = set(kept.method) if len(kept) else set()
    missing = [m for m in WANTED if m not in have]
    if not missing:
        return kept

    # Align new methods to whatever threshold grid the cache already uses.
    thr = sorted(set(kept.snr_threshold.astype(float)) | need) if len(kept) else sorted(need)
    print(f"  [{code}] computing reference methods {missing} (tobs={tobs}, dt={dt}, tdi={tdi}) ...")

    def mkrow(method, t, cnt):
        return {"code": code, "method": method, "tdi": tdi, "tobs_yr": tobs,
                "dt_s": dt, "snr_threshold": float(t), "ref_count": cnt}

    new_rows = []
    if "gbgpu" in missing:
        out = reference_snr.reference_counts(input_cat, tobs, dt, tdi=tdi, thresholds=thr,
                                             use_gpu=use_gpu, alldwds=alldwds, methods=("gbgpu",))
        new_rows += [mkrow("gbgpu", t, cnt) for t, cnt in out.get("gbgpu", {}).items()]
    if "legwork" in missing:
        out = reference_snr.reference_counts(input_cat, tobs, dt, tdi=tdi, thresholds=thr,
                                             use_gpu=use_gpu, alldwds=alldwds, methods=("legwork",))
        new_rows += [mkrow("legwork", t, cnt) for t, cnt in out.get("legwork", {}).items()]
    if "legwork_robson19" in missing:
        out = reference_snr.reference_counts(input_cat, tobs, dt, tdi=tdi, thresholds=thr,
                                             use_gpu=use_gpu, alldwds=alldwds, methods=("legwork",),
                                             legwork_confusion="robson19")
        new_rows += [mkrow("legwork_robson19", t, cnt) for t, cnt in out.get("legwork", {}).items()]

    if not new_rows:
        return kept if len(kept) else None
    merged = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True) if len(kept) else pd.DataFrame(new_rows)
    merged.to_csv(cache_path, index=False)
    print(f"  [{code}] reference cached -> {cache_path}")
    return merged


def _nominal_and_err(sub, valcol):
    """(nominal at MONEY_CUTOFF, low_err, high_err) over the cutoff sweep.

    Bar height = value at SNR>MONEY_CUTOFF; error bar spans min..max over cutoffs.
    """
    if sub.empty:
        return np.nan, 0.0, 0.0
    by_cut = sub.dropna(subset=[valcol]).set_index("cutoff")[valcol]
    if by_cut.empty:
        return np.nan, 0.0, 0.0
    nominal = by_cut.get(MONEY_CUTOFF, by_cut.sort_index().iloc[len(by_cut) // 2])
    lo, hi = by_cut.min(), by_cut.max()
    return nominal, max(nominal - lo, 0.0), max(hi - nominal, 0.0)


# Font sizes tuned for this dense layout (8 codes × 6 ICVs at figsize 10×5). Applied via
# rc_context so the figure is sized consistently regardless of ambient rcParams (e.g. the
# 28/32 usetex sizes apply_global_plot_settings leaves active after main_loop). usetex /
# font family are left untouched, so the house style still renders when it is on.
_FONT_RC = {
    "font.size": 12,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}


def _icv_bar_plot(tab, valcol, ylabel, outfile, ylim=None, legend=True):
    """Per-code grouped bars colored by ICV (lisa_dwd_count_plotter convention), with
    SNR-threshold-range error bars (bar at SNR>MONEY_CUTOFF, caps span cut min..max).

    legend=False omits the ICV legend (the recovery figures reuse the counts legend
    when placed side by side in the draft)."""
    width = 0.7 / len(VARIATIONS)
    colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(VARIATIONS)))
    xtick_pos = np.linspace((len(VARIATIONS) / 2 - 0.5) * width,
                            len(CODES) - 1 + (len(VARIATIONS) / 2 - 0.5) * width, len(CODES))
    with plt.rc_context(_FONT_RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, code in enumerate(CODES):
            for j, var in enumerate(VARIATIONS):
                sub = tab[(tab.code == code) & (tab.variation == var)]
                val, elo, ehi = _nominal_and_err(sub, valcol)
                yerr = [[elo], [ehi]] if not np.isnan(val) else None
                ax.bar(i + j * width, val, width, color=colors[j], edgecolor="black",
                       yerr=yerr, capsize=3, ecolor="black", error_kw={"elinewidth": 0.8})
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(CODES)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        if legend:
            ax.legend(VARIATIONS)   # lisa_dwd_count_plotter convention: legend(var_list) — full ICV colour key
        ax.grid(True, linestyle=":", linewidth=1.0, axis="y")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=10)
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig(os.path.join("figures", outfile), dpi=150)
        print(f"saved figures/{outfile}")
        plt.close(fig)


def _fmt(sub, valcol, pct=False):
    """'nominal (+hi/-lo)' string: value at SNR>MONEY_CUTOFF with the cutoff-sweep range."""
    nominal, elo, ehi = _nominal_and_err(sub, valcol)
    if np.isnan(nominal):
        return "n/a"
    d = 1 if pct else 0
    suffix = "%" if pct else ""
    return f"{nominal:.{d}f} (+{ehi:.{d}f}/-{elo:.{d}f}){suffix}"


def main():
    os.environ.setdefault("EXPERIMENT_ROOT", "./")
    config = load_and_prepare_config("config.yaml")
    base_datapath = config["datapath"]
    use_gpu = config.get("use_gpu", False)
    ref_keys = [k for k, _, _, _ in RECOVERY_REFS]

    # Build a (code, ICV, cutoff) table from whatever runs exist under each ICV dir.
    records = []
    for var in VARIATIONS:
        var_datapath = datapath_for_variation(base_datapath, var)
        outpath = os.path.join(config["outputpath"], var_datapath)
        inpath = os.path.join(config["inputpath"], var_datapath)
        for code in CODES:
            runs = discover_runs(outpath, code)
            if not runs:
                continue
            cuts_here = sorted(c for c, _ in runs)
            ref = load_reference(code, config, outpath, inpath, var_datapath, cuts_here, use_gpu)
            for cutoff, cat_path in runs:
                res = resolved_count(cat_path)
                rec = {"code": code, "variation": var, "cutoff": cutoff, "resolved": res}
                for key in ref_keys:
                    rec[f"recovery_{key}"] = np.nan
                    if ref is not None:
                        s = ref[(ref.method == key) & (ref.snr_threshold == cutoff)]
                        if not s.empty and int(s.ref_count.iloc[0]) > 0:
                            rec[f"recovery_{key}"] = 100.0 * res / int(s.ref_count.iloc[0])
                records.append(rec)

    if not records:
        print("No runs found (expected {code}_output_cat_snr*.h5 under the ICV dirs).")
        return
    tab = pd.DataFrame(records)
    cutoffs = sorted(tab.cutoff.unique())
    icvs_present = [v for v in VARIATIONS if v in set(tab.variation)]
    avail_refs = [(k, lbl) for k, lbl, _, _ in RECOVERY_REFS if tab[f"recovery_{k}"].notna().any()]
    print(f"ICVs with output: {icvs_present}")
    print(f"snr_cutoff runs: {cutoffs}  (value = SNR>{MONEY_CUTOFF:g}; +hi/-lo spans the sweep)")

    for var in icvs_present:
        print(f"\n=== {var} ===")
        for code in CODES:
            sub = tab[(tab.code == code) & (tab.variation == var)]
            if sub.empty:
                continue
            recs = "   ".join(f"rec[{k}] = {_fmt(sub, f'recovery_{k}', pct=True)}"
                              for k, _ in avail_refs)
            print(f"  {code:8s}  res = {_fmt(sub, 'resolved')}   {recs}")

    # lisa_dwd_count_plotter convention: per-code grouped bars by ICV. Only the
    # counts figure carries the legend; the recovery figures reuse it in the draft.
    _icv_bar_plot(tab, "resolved", "Resolved DWDs", "compare_resolved.png", legend=True)
    for key, label, outfile, ylim in RECOVERY_REFS:
        if tab[f"recovery_{key}"].notna().any():
            _icv_bar_plot(tab, f"recovery_{key}", label, outfile, ylim=ylim, legend=False)


if __name__ == "__main__":
    main()
