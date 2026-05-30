"""
Recovery-fraction analysis: resolved (gwg.icloop) / no-foreground reference,
following Wouter's lisa_dwd_counter.py (`lisa_dwd_count_plotter`) convention.

Per-code grouped bars colored by ICV (initial-condition variation). The SNR-
threshold uncertainty (sweep of snr_cutoff) is shown ON TOP as error bars: each
bar is the nominal SNR>7 value; the error bar spans the cut5..cut9 range. ICVs
without output get reserved (empty) slots, matching the upstream plotter.

Figures:
    figures/compare_resolved.png   resolved DWD counts per code (bars by ICV) + SNR-range error bars
    figures/compare_recovery.png   recovery % per code (resolved / no-FG gbgpu ref) + SNR-range error bars

Per (code, ICV, cutoff): resolved from {code}_output_cat_snr{X}.h5; the no-FG
reference (gbgpu per-source + legwork) from reference_snr.py, cached per code in
each ICV's output dir and tagged by (tobs,dt,tdi). Recovery uses the gbgpu ref
(REF_METHOD). Run from Foregrounds/. First run per ICV computes the reference
(expensive); later runs read the cache.

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
REF_METHOD = "gbgpu"   # no-FG reference used for the recovery % (apples-to-apples)


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
    """Per-code no-FG reference-count DataFrame (gbgpu+legwork), cached. None if no input_cat."""
    cache_path = os.path.join(outpath, f"{code}_reference_counts.csv")
    input_cat = os.path.join(inpath, f"{code}_input_cat.h5")
    alldwds = os.path.join(config["basepath"], datapath, f"{code}_Galaxy_AllDWDs.csv")
    tobs = float(config["duration"]); dt = float(config["dt"])
    tdi = 1 if not config.get("tdi2", False) else 2
    need = {float(t) for t in thresholds}
    if not os.path.exists(input_cat):
        print(f"  [{code}] input_cat missing; skipping reference")
        return None
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(input_cat):
        cache = pd.read_csv(cache_path)
        tag_ok = len(cache) and bool(
            ((cache.tobs_yr == tobs) & (cache.dt_s == dt) & (cache.tdi == tdi)).all())
        if tag_ok and need.issubset(set(cache.snr_threshold.astype(float))):
            return cache
        print(f"  [{code}] cache stale; recomputing")
    print(f"  [{code}] computing reference (tobs={tobs}, dt={dt}, tdi={tdi}) ...")
    counts = reference_snr.reference_counts(
        input_cat, tobs, dt, tdi=tdi, thresholds=sorted(need),
        use_gpu=use_gpu, alldwds=alldwds, methods=("gbgpu", "legwork"))
    rows = [{"code": code, "method": m, "tdi": tdi, "tobs_yr": tobs, "dt_s": dt,
             "snr_threshold": float(t), "ref_count": c}
            for m, by in counts.items() for t, c in by.items()]
    cache = pd.DataFrame(rows)
    cache.to_csv(cache_path, index=False)
    print(f"  [{code}] reference cached -> {cache_path}")
    return cache


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


def _icv_bar_plot(tab, valcol, ylabel, outfile, ylim=None):
    """Per-code grouped bars colored by ICV (lisa_dwd_count_plotter convention), with
    SNR-threshold-range error bars (bar at SNR>MONEY_CUTOFF, caps span cut min..max)."""
    width = 0.7 / len(VARIATIONS)
    colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(VARIATIONS)))
    xtick_pos = np.linspace((len(VARIATIONS) / 2 - 0.5) * width,
                            len(CODES) - 1 + (len(VARIATIONS) / 2 - 0.5) * width, len(CODES))
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
    ax.legend(VARIATIONS)   # lisa_dwd_count_plotter convention: legend(var_list) — full ICV colour key
    ax.grid(True, linestyle=":", linewidth=1.0, axis="y")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=10)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", outfile), dpi=150)
    print(f"saved figures/{outfile}")
    plt.close(fig)


def main():
    os.environ.setdefault("EXPERIMENT_ROOT", "./")
    config = load_and_prepare_config("config.yaml")
    base_datapath = config["datapath"]
    use_gpu = config.get("use_gpu", False)

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
                rec = {"code": code, "variation": var, "cutoff": cutoff,
                       "resolved": res, "recovery_pct": np.nan}
                if ref is not None:
                    s = ref[(ref.method == REF_METHOD) & (ref.snr_threshold == cutoff)]
                    if not s.empty and int(s.ref_count.iloc[0]) > 0:
                        rec["recovery_pct"] = 100.0 * res / int(s.ref_count.iloc[0])
                records.append(rec)

    if not records:
        print("No runs found (expected {code}_output_cat_snr*.h5 under the ICV dirs).")
        return
    tab = pd.DataFrame(records)
    cutoffs = sorted(tab.cutoff.unique())
    icvs_present = [v for v in VARIATIONS if v in set(tab.variation)]
    print(f"ICVs with output: {icvs_present}")
    print(f"snr_cutoff runs: {cutoffs}  (bar = SNR>{MONEY_CUTOFF:g}; error bar spans the range)")

    for var in icvs_present:
        print(f"\n=== {var} ===")
        for code in CODES:
            sub = tab[(tab.code == code) & (tab.variation == var)].sort_values("cutoff")
            if sub.empty:
                continue
            cells = "  ".join(f"cut{r.cutoff:g}: res={r.resolved} rec={r.recovery_pct:.1f}%"
                              for _, r in sub.iterrows())
            print(f"  {code:8s}  {cells}")

    # Restored lisa_dwd_count_plotter convention: per-code grouped bars by ICV.
    _icv_bar_plot(tab, "resolved", "Resolved DWDs", "compare_resolved.png")
    _icv_bar_plot(tab, "recovery_pct", f"Resolved / no-FG ({REF_METHOD}) [%]",
                  "compare_recovery.png", ylim=(0, 100))


if __name__ == "__main__":
    main()
