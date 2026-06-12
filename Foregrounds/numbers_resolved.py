"""
numbers_resolved.py -- refresh the N_1kpc count grids with OUR resolved sources.

Adapted from analysis-scripts/numbers.ipynb: the hardcoded count dicts are replaced by a
computation over our pipeline output. Per (code, leaf) it counts the RESOLVED DWDs (icloop
SNR>7, from {code}_output_cat_snr7.h5) that are CLOSE (< 1 kpc, for optical follow-up) AND
in the LISA band (f > 1e-4 Hz), then renders the per-family grids
    figures/N_1kpc_grid_MT_variations.pdf      (mass-transfer subfamilies)
    figures/N_1kpc_grid_ic_variations.pdf      (initial-condition variations)
plus figures/N_1kpc_resolved_table.csv (the durable artifact).

Distance recovery (no pipeline rerun, no added index): UID is NOT unique per galaxy
instance -- the same pop-synth binary is sampled at many (time, position) combos, so UID/ID
repeat and frequency/distance vary within a UID. But the binary's MASSES are UID-invariant,
so Mc(UID) is a clean one-per-UID lookup; the position-dependent distance then comes from
each resolved source's OWN (Amplitude, Frequency) by inverting gen_catalog's amplitude
    A = 2 Mc^(5/3) (pi f)^(2/3) / d   ->   d = 2 Mc(UID)^(5/3) (pi f)^(2/3) / A
(verified exact to ~1e-14 against RRelkpc). The same UID join later yields formation channels.

Run from Foregrounds/ (needs the gbgpu env for gwg + the catalogs/output on disk):
    python numbers_resolved.py --datapath monte_carlo_comparisons/
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gwg

from helpers import Constants, load_and_prepare_config

CODES = ['BPASS', 'BSE', 'ComBinE', 'COMPAS', 'COSMIC', 'METISSE', 'SeBa', 'SEVN']
SNR = 7            # the snr7 resolved run
FCUT = 1e-4        # Hz, LISA-band lower edge ("in band")
DMAX_KPC = 1.0     # kpc, "close" cut for optical follow-up

# Variation grouping + labels, carried over verbatim from numbers.ipynb so the two grids
# match the collaborators' tables.
MT_ORDER = ['fiducial', 'qcrit_claeys_14', 'qcrit_hurley_02', 'qcrit_hurley_webbink', 'qcrit_zetas',
            'alpha_gamma_2', 'alpha_lambda_02', 'alpha_lambda_05', 'alpha_lambda_1', 'alpha_lambda_2',
            'accretion_0', 'accretion_05', 'accretion_1']
IC_ORDER = ['fiducial', 'porb_log_uniform', 'm2_min_05', 'qmin_01', 'thermal_ecc', 'uniform_ecc']
LABEL_MAP = {
    'fiducial': 'fiducial',
    'qcrit_claeys_14': r'$q_c$: Claeys+14', 'qcrit_hurley_02': r'$q_c$: Hurley+02',
    'qcrit_hurley_webbink': r'$q_c$: Hjellming+83', 'qcrit_zetas': r'$q_c$: $\zeta$',
    'alpha_gamma_2': r'$\gamma \alpha = 2$',
    'alpha_lambda_02': r'$\alpha \lambda = 0.2$', 'alpha_lambda_05': r'$\alpha \lambda = 0.5$',
    'alpha_lambda_1': r'$\alpha \lambda = 1$', 'alpha_lambda_2': r'$\alpha \lambda = 2$',
    'accretion_0': r'$\beta = 0$', 'accretion_05': r'$\beta = 0.5$', 'accretion_1': r'$\beta = 1$',
    'porb_log_uniform': r'$P_{\mathrm{orb}}$: log-uniform', 'm2_min_05': r'$m_{2,\min}=0.5$',
    'qmin_01': r'$q_{\min}=0.1$', 'thermal_ecc': 'thermal ecc', 'uniform_ecc': 'uniform ecc',
}


def discover_leaves(outputpath, mc_prefix):
    """Every leaf with a {code}_output_cat_snr{SNR}.h5 under outputpath/mc_prefix, as
    [(datapath_rel, family, variation)] (ICV flat + MTV nested), mirroring compare_resolved."""
    root = os.path.join(outputpath, mc_prefix)
    leaves = {}
    for cat in glob.glob(os.path.join(root, "**", f"*_output_cat_snr{SNR}.h5"), recursive=True):
        rel = os.path.relpath(os.path.dirname(cat), outputpath)
        if rel in leaves:
            continue
        parts = rel.split(os.sep)
        family = ("initial_condition_variations" if "initial_condition_variations" in parts
                  else "mass_transfer_variations" if "mass_transfer_variations" in parts else "other")
        leaves[rel] = (rel + os.sep, family, parts[-1])
    return sorted(leaves.values())


def uid_chirpmass(alldwds_csv):
    """UID -> chirp mass [s] from the source catalog (masses are UID-invariant; dedup by UID)."""
    df = pd.read_csv(alldwds_csv, usecols=['UID', 'mass1', 'mass2']).drop_duplicates('UID')
    m1 = df.mass1.values * Constants.Msun
    m2 = df.mass2.values * Constants.Msun
    Mc = (m1 + m2) * (m1 * m2 / (m1 + m2) ** 2) ** (3. / 5)
    return pd.Series(Mc, index=df.UID.values)


def distance_kpc(uid, f, A, mc_of_uid):
    """Per-source heliocentric distance [kpc] from its own (f, A) + Mc(UID), inverting
    A = 2 Mc^(5/3)(pi f)^(2/3)/d (gen_catalog convention; d in s -> kpc via Constants.pc*1e3)."""
    Mc = mc_of_uid.reindex(uid).values            # NaN if a resolved UID is absent (shouldn't happen)
    d_s = 2.0 * Mc ** (5. / 3) * (np.pi * f) ** (2. / 3) / A
    return d_s / (Constants.pc * 1e3)


def count_close_inband(alldwds_csv, out_h5):
    """# resolved DWDs with f > FCUT and d < DMAX_KPC, or NaN if the run is absent."""
    if not os.path.exists(out_h5):
        return np.nan
    cat = gwg.utils.load_h5(out_h5, key='cat')
    uid = np.asarray(cat['Name'])
    f = np.asarray(cat['Frequency'])
    A = np.asarray(cat['Amplitude'])
    d = distance_kpc(uid, f, A, uid_chirpmass(alldwds_csv))
    return int(np.sum((f > FCUT) & (d < DMAX_KPC)))


def color_edges(values):
    """Two integer edges (33rd/66th pct of the positive counts) so the 4-colour scheme
    auto-scales to the resolved counts (far smaller than the catalog N_1kpc the notebook
    coloured), shared across both grids for comparability."""
    pos = np.array([v for v in values if np.isfinite(v) and v > 0], dtype=float)
    if pos.size == 0:
        return 1, 2
    e1, e2 = np.percentile(pos, [33, 66])
    e1, e2 = int(round(e1)), int(round(e2))
    return (e1, max(e2, e1 + 1))


def _cell_color(val, e1, e2):
    if pd.isna(val):
        return "white"
    if val == 0:
        return "#d9d9d9"
    if val < e1:
        return "#f8d7da"
    if val < e2:
        return "#fff3cd"
    return "#d4edda"


def draw_grid(full_df, var_order, outfile, e1, e2):
    """Grid figure (rows=code, cols=variation) of the resolved close+in-band counts, in the
    numbers.ipynb style. Missing (code,var) -> em-dash; 0 -> '-'."""
    pivot = full_df.pivot(index="code", columns="variation", values="N")
    var_order = [v for v in var_order if v in pivot.columns]
    if not var_order:
        print(f"  (no variations present for {outfile}; skipped)")
        return
    pivot = pivot[var_order]
    var_labels = [LABEL_MAP.get(v, v) for v in var_order]
    codes = pivot.index.tolist()
    variations = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(len(variations) * 1.3, len(codes) * 0.7))
    ax.set_aspect('equal')
    ax.axis('off')
    for i, code in enumerate(codes):
        for j, var in enumerate(variations):
            val = pivot.loc[code, var]
            text = "—" if pd.isna(val) else ("-" if val == 0 else f"{int(val):,}")
            rect = plt.Rectangle([j, len(codes) - i - 1], 1, 1,
                                 facecolor=_cell_color(val, e1, e2), edgecolor='#cccccc', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, len(codes) - i - 0.5, text, ha='center', va='center', fontsize=10,
                    color='black' if not pd.isna(val) else '#aaaaaa')
    for j, label in enumerate(var_labels):
        ax.text(j + 0.3, len(codes) + 0.1, label, ha="left", va="bottom", fontsize=10,
                rotation=45, rotation_mode="anchor")
    for i, code in enumerate(codes):
        ax.text(-0.1, len(codes) - i - 0.5, code, ha='right', va='center', fontsize=10)

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="#cccccc", label="0"),
        Patch(facecolor="#f8d7da", edgecolor="#cccccc", label=f"< {e1:,}"),
        Patch(facecolor="#fff3cd", edgecolor="#cccccc", label=f"{e1:,}-{e2 - 1:,}"),
        Patch(facecolor="#d4edda", edgecolor="#cccccc", label=f">= {e2:,}"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0, -0.02),
              ncol=4, frameon=False, fontsize=9, title="resolved DWDs < 1 kpc, f > 0.1 mHz")
    ax.set_xlim(-2, len(variations))
    ax.set_ylim(-0.5, len(codes) + 2.5)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", outfile), bbox_inches='tight')
    print(f"saved figures/{outfile}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datapath", required=True,
                    help="any catalog subpath in the tree; only its top component "
                         "(monte_carlo_comparisons[_lightweight_500K_DWDs]) selects the tree to scan.")
    args = ap.parse_args()
    os.environ.setdefault("EXPERIMENT_ROOT", "./")
    config = load_and_prepare_config("config.yaml")
    mc_prefix = args.datapath.rstrip("/").split("/")[0]
    leaves = discover_leaves(config["outputpath"], mc_prefix)
    if not leaves:
        print(f"No resolved runs found ({{code}}_output_cat_snr{SNR}.h5) under "
              f"{os.path.join(config['outputpath'], mc_prefix)}.")
        return

    rows = []
    for datapath_rel, family, variation in leaves:
        for code in CODES:
            out_h5 = os.path.join(config["outputpath"], datapath_rel, f"{code}_output_cat_snr{SNR}.h5")
            if not os.path.exists(out_h5):
                continue
            alldwds = os.path.join(config["basepath"], datapath_rel, f"{code}_Galaxy_AllDWDs.csv")
            n = count_close_inband(alldwds, out_h5)
            rows.append({"code": code, "family": family, "variation": variation, "N": n})
            print(f"  {family}/{variation:24s} {code:8s}  N(<1kpc, f>1e-4) = {n}")

    if not rows:
        print("No (code, leaf) resolved catalogs to count.")
        return
    full_df = pd.DataFrame(rows)
    os.makedirs("figures", exist_ok=True)
    full_df.sort_values(["family", "variation", "code"]).to_csv(
        os.path.join("figures", "N_1kpc_resolved_table.csv"), index=False)
    print(f"\nsaved figures/N_1kpc_resolved_table.csv ({len(full_df)} rows)")

    e1, e2 = color_edges(full_df.N.values)
    print(f"colour edges (33/66 pct of positive counts): {e1}, {e2}")
    draw_grid(full_df[full_df.family == "mass_transfer_variations"], MT_ORDER,
              "N_1kpc_grid_MT_variations.pdf", e1, e2)
    draw_grid(full_df[full_df.family == "initial_condition_variations"], IC_ORDER,
              "N_1kpc_grid_ic_variations.pdf", e1, e2)


if __name__ == "__main__":
    main()
