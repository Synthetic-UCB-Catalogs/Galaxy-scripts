"""
Recovery-fraction analysis: resolved (gwg.icloop) / no-foreground reference.

For the datapath in config.yaml, this:
  1. Discovers per-snr_cutoff pipeline runs ({code}_output_cat_snrX.h5, written
     by main_loop.py), reading each run's snr_cutoff from its saved run_config.
  2. Loads-or-computes the no-FG reference counts via reference_snr.py
     (gbgpu per-source + legwork sky-averaged), cached per code in the output
     folder and tagged by (tobs, dt, tdi). The cache is recomputed if absent,
     if the tag mismatches, if a needed threshold is missing, or if input_cat.h5
     is newer than the cache (e.g. after an amplitude/gen_catalog change).
  3. Plots:
       figures/resolved_counts.png      resolved DWDs per code, grouped bars by
                                        cutoff (gist_rainbow; lisa_dwd_count_plotter
                                        convention) — the main counts figure
       figures/recovery_money.png       recovery at SNR>7 per code
                                        (gbgpu = headline, legwork = faded check)
       figures/recovery_vs_cutoff.png   recovery vs snr_cutoff per code (gbgpu)

Recovery(code, cutoff X) = resolved_count(run X) / reference[method](SNR>X).

Scope: one datapath (the one in config.yaml) per invocation. Re-point config's
datapath (or loop externally) for other ICVs. Run from Foregrounds/.
First run is expensive (computes the reference); later runs read the cache.

Usage:
    python compare_resolved.py
"""
import os
import re
import glob

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import gwg

from helpers import load_and_prepare_config
import reference_snr

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']
MONEY_CUTOFF = 7.0


def _apply_axes_style(ax):
    ax.grid(True, which="major", linestyle=":", linewidth=1.0)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=8)


def discover_runs(outpath, code):
    """[(cutoff, cat_path, cfg_path)] for each snr-cutoff run of `code`."""
    runs = []
    for cat_path in sorted(glob.glob(os.path.join(outpath, f"{code}_output_cat_snr*.h5"))):
        m = re.search(r"_output_cat_snr([0-9.]+)\.h5$", os.path.basename(cat_path))
        if not m:
            continue
        tag = m.group(1)
        cfg_path = os.path.join(outpath, f"{code}_run_config_snr{tag}.yaml")
        runs.append((float(tag), cat_path, cfg_path))
    return runs


def resolved_count(cat_path):
    return len(gwg.utils.load_h5(cat_path, key="cat"))


def load_reference(code, config, outpath, inpath, datapath, thresholds, use_gpu):
    """Return the per-code reference-count DataFrame, from cache or fresh."""
    cache_path = os.path.join(outpath, f"{code}_reference_counts.csv")
    input_cat = os.path.join(inpath, f"{code}_input_cat.h5")
    alldwds = os.path.join(config["basepath"], datapath, f"{code}_Galaxy_AllDWDs.csv")
    tobs = float(config["duration"])
    dt = float(config["dt"])
    tdi = 1 if not config.get("tdi2", False) else 2
    need = {float(t) for t in thresholds}

    if not os.path.exists(input_cat):
        print(f"  [{code}] input_cat missing ({input_cat}); skipping")
        return None

    if os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(input_cat):
        cache = pd.read_csv(cache_path)
        tag_ok = len(cache) and bool(
            ((cache.tobs_yr == tobs) & (cache.dt_s == dt) & (cache.tdi == tdi)).all())
        have = set(cache.snr_threshold.astype(float))
        if tag_ok and need.issubset(have):
            return cache
        print(f"  [{code}] cache stale (tag/threshold mismatch); recomputing")

    print(f"  [{code}] computing reference (tobs={tobs}, dt={dt}, tdi={tdi}) ...")
    counts = reference_snr.reference_counts(
        input_cat, tobs, dt, tdi=tdi, thresholds=sorted(need),
        use_gpu=use_gpu, alldwds=alldwds, methods=("gbgpu", "legwork"))
    rows = [
        {"code": code, "method": method, "tdi": tdi, "tobs_yr": tobs,
         "dt_s": dt, "snr_threshold": float(thr), "ref_count": c}
        for method, by_thr in counts.items() for thr, c in by_thr.items()
    ]
    cache = pd.DataFrame(rows)
    cache.to_csv(cache_path, index=False)
    print(f"  [{code}] reference cached -> {cache_path}")
    return cache


def main():
    os.environ.setdefault("EXPERIMENT_ROOT", "./")
    config = load_and_prepare_config("config.yaml")
    datapath = config["datapath"]
    outpath = os.path.join(config["outputpath"], datapath)
    inpath = os.path.join(config["inputpath"], datapath)
    use_gpu = config.get("use_gpu", False)

    # 1. Discover runs and the set of cutoffs present (= reference thresholds needed).
    runs_by_code = {c: discover_runs(outpath, c) for c in CODES}
    cutoffs = sorted({cut for runs in runs_by_code.values() for cut, _, _ in runs})
    if not cutoffs:
        print(f"No runs found under {outpath} (expected {{code}}_output_cat_snr*.h5).")
        return
    print(f"datapath: {datapath}")
    print(f"snr_cutoff runs found: {cutoffs}")

    # 2. Build the recovery table: one row per (code, cutoff, method).
    records = []
    for code in CODES:
        runs = runs_by_code[code]
        if not runs:
            continue
        ref = load_reference(code, config, outpath, inpath, datapath, cutoffs, use_gpu)
        if ref is None:
            continue
        for cutoff, cat_path, _cfg in runs:
            res = resolved_count(cat_path)
            for method in ref.method.unique():
                sub = ref[(ref.method == method) & (ref.snr_threshold == cutoff)]
                if sub.empty:
                    continue
                ref_n = int(sub.ref_count.iloc[0])
                rec = 100.0 * res / ref_n if ref_n > 0 else np.nan
                records.append({"code": code, "cutoff": cutoff, "method": method,
                                "resolved": res, "ref_count": ref_n, "recovery_pct": rec})
    if not records:
        print("No reference/resolved pairs to plot.")
        return
    tab = pd.DataFrame(records)

    print("\n=== recovery table ===")
    for code in CODES:
        sub = tab[tab.code == code]
        if sub.empty:
            continue
        print(f"\n{code}:")
        for _, r in sub.sort_values(["cutoff", "method"]).iterrows():
            print(f"  cutoff {r.cutoff:g}  {r.method:7s}  resolved={r.resolved:>7d}  "
                  f"ref={r.ref_count:>7d}  recovery={r.recovery_pct:5.1f}%")

    os.makedirs("figures", exist_ok=True)

    # 3a. Money figure: recovery at SNR>7 per code (gbgpu headline, legwork faded).
    money = tab[tab.cutoff == MONEY_CUTOFF]
    if not money.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        present = [c for c in CODES if c in set(money.code)]
        x = np.arange(len(present))
        w = 0.38
        for method, dx, alpha, lbl in [("gbgpu", -w / 2, 1.0, "gbgpu-ref (per-source)"),
                                       ("legwork", w / 2, 0.4, "legwork-ref (sky-avg)")]:
            vals = [money[(money.code == c) & (money.method == method)].recovery_pct.mean()
                    for c in present]
            ax.bar(x + dx, vals, w, alpha=alpha, edgecolor="black", label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(present)
        ax.set_ylabel(r"Recovery at SNR$>$7  [\%]" if plt.rcParams.get("text.usetex")
                      else "Recovery at SNR>7 [%]")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")
        _apply_axes_style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "recovery_money.png"), dpi=150)
        print("\nsaved figures/recovery_money.png")

    # 3b. Uncertainty figure: recovery vs snr_cutoff per code (gbgpu reference).
    gb = tab[tab.method == "gbgpu"]
    if not gb.empty and len(cutoffs) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for code in CODES:
            sub = gb[gb.code == code].sort_values("cutoff")
            if sub.empty:
                continue
            ax.plot(sub.cutoff, sub.recovery_pct, marker="o", label=code)
        ax.set_xlabel("SNR cutoff")
        ax.set_ylabel("Recovery [%] (gbgpu reference)")
        ax.set_ylim(0, 100)
        ax.legend(loc="best", ncol=2, fontsize=9)
        _apply_axes_style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "recovery_vs_cutoff.png"), dpi=150)
        print("saved figures/recovery_vs_cutoff.png")
    elif len(cutoffs) <= 1:
        print("(only one snr_cutoff present — run more cutoffs for the uncertainty figure)")

    # 3c. Main resolved-COUNTS bar plot. Restores the upstream lisa_dwd_count_plotter
    # convention (per-code grouped bars, gist_rainbow per group, centered xticks, dotted
    # y-grid, inward ticks) — here grouped by SNR cutoff (what we now sweep) rather than
    # by ICV. Shows the pipeline's resolved counts directly.
    present = [c for c in CODES if c in set(tab.code)]
    if present and cutoffs:
        width = 0.7 / len(cutoffs)
        colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(cutoffs)))
        xtick_pos = np.linspace(
            (len(cutoffs) / 2 - 0.5) * width,
            len(present) - 1 + (len(cutoffs) / 2 - 0.5) * width,
            len(present),
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        labeled = set()
        for i, code in enumerate(present):
            for j, cut in enumerate(cutoffs):
                sub = tab[(tab.code == code) & (tab.cutoff == cut)]
                val = int(sub.resolved.iloc[0]) if not sub.empty else np.nan
                lbl = f"SNR>{cut:g}" if (cut not in labeled and not sub.empty) else None
                ax.bar(i + j * width, val, width, color=colors[j],
                       edgecolor="black", label=lbl)
                if lbl:
                    labeled.add(cut)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(present)
        ax.set_ylabel("Resolved DWDs")
        ax.legend(title="cutoff")
        ax.grid(True, linestyle=":", linewidth=1.0, axis="y")
        ax.yaxis.set_ticks_position("both")
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=10)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", "resolved_counts.png"), dpi=150)
        print("saved figures/resolved_counts.png")


if __name__ == "__main__":
    main()
