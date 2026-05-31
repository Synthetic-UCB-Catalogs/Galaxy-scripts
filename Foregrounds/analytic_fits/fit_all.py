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
