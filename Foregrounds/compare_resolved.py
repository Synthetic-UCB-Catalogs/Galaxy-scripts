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
import argparse

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


def discover_leaves(outputpath, mc_prefix):
    """Discover every catalog leaf under outputpath/mc_prefix that holds at least one
    {code}_output_cat_snr*.h5, as [(datapath_rel, family, variation)].

    Generalizes the old hardcoded 6-ICV loop to whatever is on disk, so both the flat
    initial_condition_variations/{var}/ layout and the nested
    mass_transfer_variations/{subfamily}/{var}/ layout are picked up. `family` is the
    taxonomy folder; `variation` is the leaf dir name. The same variation name can occur
    under both families (e.g. a 'fiducial' in each), so callers key on (family, variation).
    """
    root = os.path.join(outputpath, mc_prefix)
    leaves = {}
    for cat in glob.glob(os.path.join(root, "**", "*_output_cat_snr*.h5"), recursive=True):
        leafdir = os.path.dirname(cat)
        rel = os.path.relpath(leafdir, outputpath)
        if rel in leaves:
            continue
        parts = rel.split(os.sep)
        if "initial_condition_variations" in parts:
            family = "initial_condition_variations"
        elif "mass_transfer_variations" in parts:
            family = "mass_transfer_variations"
        else:
            family = "other"
        leaves[rel] = (rel + os.sep, family, parts[-1])
    return sorted(leaves.values())


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


def _icv_noise_curves(config, channel="A"):
    """Per-code overlay of the final smoothed noise curve S (channel `channel`) across ICVs,
    coloured by ICV with the same gist_rainbow + legend(VARIATIONS) scheme as the recovery
    bar plots (reserved slots for missing ICVs) + the TDI1 instrument reference. No analytic
    fits. Two figures per code:
      figures/noise_curves_{code}.png       central (SNR>MONEY_CUTOFF) curve per ICV
      figures/noise_curves_band_{code}.png  + SNR-threshold band shaded between snr5 and snr9
    (lower cutoff subtracts more sources -> lower residual, so the band brackets the curve;
    same cut5..cut9 span the bar plot shows as error bars). The band needs both the snr5 and
    snr9 runs for that ICV; ICVs missing either just get the central curve.
    """
    base = config["datapath"]
    colors = plt.get_cmap("gist_rainbow")(np.linspace(0, 1, len(VARIATIONS)))
    tdi = 1 if not config.get("tdi2", False) else 2
    fref = np.logspace(-4, np.log10(2e-2), 2000)
    instr = np.asarray(reference_snr._instrument_psd_fn(tdi)(fref)[0])
    cuts = (5.0, MONEY_CUTOFF, 9.0)
    os.makedirs("figures", exist_ok=True)

    def _load(h5):
        S = gwg.utils.load_h5(h5, key="S")
        fa = np.asarray(S["f"]); Sa = np.abs(np.asarray(S[channel]))
        m = (fa >= 5e-5) & (fa <= 2.5e-2)
        fa, Sa = fa[m], Sa[m]
        step = max(1, len(fa) // 5000)                # the curve is smooth; downsample for speed
        return fa[::step], Sa[::step]

    for code in CODES:
        per_icv = {}                                  # j -> (fa, {cutoff: Sa}); fa shared across cutoffs
        for j, var in enumerate(VARIATIONS):
            outpath = os.path.join(config["outputpath"], datapath_for_variation(base, var))
            fa, byc = None, {}
            for cut in cuts:
                h5 = os.path.join(outpath, f"{code}_output_cat_snr{cut:g}.h5")
                if not os.path.exists(h5):
                    continue
                try:
                    fa, byc[cut] = _load(h5)
                except Exception:
                    continue
            if byc:
                per_icv[j] = (fa, byc)
        if not per_icv:
            continue
        for kind in ("lines", "band"):
            with plt.rc_context(_FONT_RC):
                fig, ax = plt.subplots(figsize=(10, 5))
                ymax = float(instr.min())
                for j, var in enumerate(VARIATIONS):
                    if j not in per_icv:
                        ax.loglog([], [], color=colors[j], label=var)   # reserved slot (mirror bar plot)
                        continue
                    fa, byc = per_icv[j]
                    central = byc.get(MONEY_CUTOFF, byc[sorted(byc)[len(byc) // 2]])
                    ax.loglog(fa, central, color=colors[j], lw=1.5, label=var)
                    iv = (fa >= 1e-4) & (fa <= 2e-2)
                    if iv.any():
                        ymax = max(ymax, float(np.nanmax(central[iv])))
                    if kind == "band" and 5.0 in byc and 9.0 in byc:
                        ax.fill_between(fa, byc[5.0], byc[9.0], color=colors[j], alpha=0.25, lw=0)
                ax.loglog(fref, instr, "k--", lw=1.2, alpha=0.6, label="instrument")
                ax.set_xlim(1e-4, 2e-2)
                ax.set_ylim(max(float(instr.min()) * 0.5, 1e-44), ymax * 2)
                ax.set_xlabel(r"Frequency [Hz]")
                ax.set_ylabel(r"PSD [1/Hz]")
                ax.legend()                            # labelled lines only -> ICV colour key + instrument
                ax.text(0.98, 0.98, code, transform=ax.transAxes, ha="right", va="top", fontsize=14)
                ax.grid(True, linestyle=":", linewidth=1.0)
                ax.xaxis.set_ticks_position("both")
                ax.yaxis.set_ticks_position("both")
                ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=10)
                fig.tight_layout()
                suffix = "" if kind == "lines" else "_band"
                out = os.path.join("figures", f"noise_curves{suffix}_{code}.png")
                fig.savefig(out, dpi=150)
                print(f"saved {out}")
                plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", required=True,
                    help="any catalog subpath in the tree to analyze; only its top component "
                         "(monte_carlo_comparisons[_lightweight_500K_DWDs]) is used to scope "
                         "discovery. REQUIRED; no config.yaml default, matching the pipeline scripts.")
    args = ap.parse_args()
    os.environ.setdefault("EXPERIMENT_ROOT", "./")
    config = load_and_prepare_config("config.yaml")
    base_datapath = args.datapath
    use_gpu = config.get("use_gpu", False)
    ref_keys = [k for k, _, _, _ in RECOVERY_REFS]

    # Discover EVERY catalog leaf with output under the active monte_carlo_comparisons[
    # _lightweight] tree — ICV (flat) and MTV (nested) alike — instead of looping the
    # hardcoded 6-ICV list. Build a (code, family, variation, cutoff) recovery table.
    mc_prefix = base_datapath.rstrip("/").split("/")[0]
    leaves = discover_leaves(config["outputpath"], mc_prefix)
    records = []
    for var_datapath, family, variation in leaves:
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
                rec = {"code": code, "family": family, "variation": variation,
                       "datapath": var_datapath, "cutoff": cutoff, "resolved": res}
                for key in ref_keys:
                    rec[f"recovery_{key}"] = np.nan
                    if ref is not None:
                        s = ref[(ref.method == key) & (ref.snr_threshold == cutoff)]
                        if not s.empty and int(s.ref_count.iloc[0]) > 0:
                            rec[f"recovery_{key}"] = 100.0 * res / int(s.ref_count.iloc[0])
                records.append(rec)

    if not records:
        print(f"No runs found (expected {{code}}_output_cat_snr*.h5 under "
              f"{os.path.join(config['outputpath'], mc_prefix)}).")
        return
    tab = pd.DataFrame(records)
    cutoffs = sorted(tab.cutoff.unique())
    avail_refs = [(k, lbl) for k, lbl, _, _ in RECOVERY_REFS if tab[f"recovery_{k}"].notna().any()]
    print(f"snr_cutoff runs: {cutoffs}  (value = SNR>{MONEY_CUTOFF:g}; +hi/-lo spans the sweep)")

    # Numbers for every leaf (ICV + MTV), grouped by family then variation.
    for family in sorted(tab.family.unique()):
        for variation in sorted(tab[tab.family == family].variation.unique()):
            print(f"\n=== {family}/{variation} ===")
            for code in CODES:
                sub = tab[(tab.code == code) & (tab.family == family) & (tab.variation == variation)]
                if sub.empty:
                    continue
                recs = "   ".join(f"rec[{k}] = {_fmt(sub, f'recovery_{k}', pct=True)}"
                                  for k, _ in avail_refs)
                print(f"  {code:8s}  res = {_fmt(sub, 'resolved')}   {recs}")

    # Full per-leaf recovery table to CSV (the durable artifact; MTV plotting deferred).
    os.makedirs("figures", exist_ok=True)
    csv_path = os.path.join("figures", "recovery_table.csv")
    tab.sort_values(["family", "variation", "code", "cutoff"]).to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}  ({len(tab)} rows, families: {sorted(tab.family.unique())})")

    # ICV grouped bar plots (lisa_dwd_count_plotter convention): _icv_bar_plot / _icv_noise_curves
    # iterate the VARIATIONS ICV list, so MTV rows are naturally excluded. MTV plotting is deferred
    # (numbers + CSV only for now); the bars below render only when ICV output is present.
    icv_tab = tab[tab.family == "initial_condition_variations"]
    if not icv_tab.empty:
        _icv_bar_plot(icv_tab, "resolved", "Resolved DWDs", "compare_resolved.png", legend=True)
        for key, label, outfile, ylim in RECOVERY_REFS:
            if icv_tab[f"recovery_{key}"].notna().any():
                _icv_bar_plot(icv_tab, f"recovery_{key}", label, outfile, ylim=ylim, legend=False)
        # _icv_noise_curves derives each ICV path from config["datapath"] (last component
        # swapped per ICV); anchor it on a discovered ICV leaf so it works regardless of
        # what config.yaml's datapath currently points at.
        config["datapath"] = icv_tab.iloc[0]["datapath"]
        _icv_noise_curves(config)
    mtv_present = sorted(tab[tab.family == "mass_transfer_variations"].variation.unique())
    if mtv_present:
        print(f"MTV leaves with output (numbers + CSV only, grouped plots deferred): {mtv_present}")


if __name__ == "__main__":
    main()
