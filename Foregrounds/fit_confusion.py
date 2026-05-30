"""
Fit an analytic galactic-confusion model to the pipeline's final foreground curve.

Reads a `{code}_output_cat_snr{X}.h5` written by main_loop (gwg.utils.to_h5):
  - key "tdi": the final RESIDUAL TDI A/E/T (complex) after resolved-source subtraction
  - key "S":   the final smoothed total PSD (mean-smoothed; = confusion + instrument)

main_loop smooths with a running MEAN (config `methoduse: mean`), which leaves spikes
from high-SNR residual sources. For the fit we re-smooth the residual with a running
MEDIAN (gwg's `methoduse: median` path: ndimage.median_filter x chi^2 debias norm),
add the instrument PSD, and fit

    S_total(f) = S_instr(f) + S_conf(f; theta)

with S_instr fixed (TDI1 AET1SensitivityMatrix, the same the pipeline adds) and S_conf the
analytic confusion model. This is preprocessing only -- no icloop rerun.

Models:
  karnesis (default, the gwg-lineage form, Karnesis+2021 Eq.6):
    S_conf(f) = (A/2) f^(-7/3) exp[ -(f/f1)^alpha ] [ 1 + tanh( (fknee - f)/f2 ) ]
    theta = {A, alpha, f1, f2, fknee}
  robson (Robson+2019 Eq.14, the form LEGWORK exposes):
    S_conf(f) = A f^(-7/3) exp[ -f^alpha + beta f sin(kappa f) ] [ 1 + tanh( gamma (fk - f) ) ]
    theta = {A, alpha, beta, kappa, gamma, fk}

NOTE on convention: we fit the pipeline's TDI1 A/E-channel PSD (1/Hz). The fitted amplitude
A is therefore in that convention, NOT Robson's sky-averaged single-Michelson convention --
the shape parameters are comparable, A is not directly. Document this with any data product.

CLI (run from Foregrounds/):
    python fit_confusion.py --h5 output/<DP>/COSMIC_output_cat_snr7.h5 --code COSMIC \
        --channels A,E --nwalkers 32 --nsteps 4000 --out fits/
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# chi^2 debias factor between the median and the mean of the periodogram (gwg's value).
MEDIAN_NORM = 1.0 / 0.7023319615912207
CHANNELS_WITH_CONFUSION = ("A", "E")   # T carries negligible galactic confusion


# --------------------------------------------------------------------------- I/O
def load_residual_tdi(h5_path):
    """Return (f, {A,E,T: complex residual}) from the 'tdi' key."""
    df = pd.read_hdf(h5_path, key="tdi")
    f = np.asarray(df["f"].values, dtype=np.float64)
    aet = {k: np.asarray(df[k].values) for k in ("A", "E", "T")}
    return f, aet


def load_saved_psd(h5_path):
    """Return (f, {A,E,T: real PSD}) from the 'S' key (the mean-smoothed pipeline curve)."""
    df = pd.read_hdf(h5_path, key="S")
    f = np.asarray(df["f"].values, dtype=np.float64)
    S = {k: np.real(np.asarray(df[k].values)).astype(np.float64) for k in ("A", "E", "T")}
    return f, S


def instrument_aet(f):
    """TDI1 AET instrument PSD on grid f (>0), matching main_loop's lisa_noise."""
    from lisatools.sensitivity import AET1SensitivityMatrix
    import lisatools.detector as lisa_models

    fpos = np.maximum(f, 1e-7)
    sm = AET1SensitivityMatrix(fpos, model=lisa_models.scirdv1, return_type="PSD")
    return {k: np.asarray(sm[i]).astype(np.float64) for i, k in enumerate(("A", "E", "T"))}


# ------------------------------------------------------------------- smoothing
def median_total_psd(f, aet, instr, window, fmax_crop=2e-2):
    """Running-median total PSD = median(periodogram)*norm + instrument, per channel.

    Crops to f <= fmax_crop before median-filtering (the confusion lives well below
    20 mHz) so the filter runs over ~2M points instead of the full Nyquist array.
    """
    df = np.abs(f[1] - f[0])
    crop = f <= fmax_crop
    fc = f[crop]
    Smed = {}
    for k in ("A", "E", "T"):
        periodogram = 2.0 * df * np.abs(aet[k][crop]) ** 2
        smoothed = ndimage.median_filter(periodogram, size=int(window)) * MEDIAN_NORM
        Smed[k] = smoothed + instr[k][crop]
    return fc, Smed


# ---------------------------------------------------------------------- models
def model_conf(f, theta, model):
    """Analytic confusion PSD S_conf(f; theta)."""
    if model == "karnesis":
        A, alpha, f1, f2, fknee = theta
        return (A / 2.0) * f ** (-7.0 / 3.0) * np.exp(-(f / f1) ** alpha) \
            * (1.0 + np.tanh((fknee - f) / f2))
    if model == "robson":
        A, alpha, beta, kappa, gamma, fk = theta
        return A * f ** (-7.0 / 3.0) * np.exp(-(f ** alpha) + beta * f * np.sin(kappa * f)) \
            * (1.0 + np.tanh(gamma * (fk - f)))
    raise ValueError(f"unknown model {model!r}")


# Karnesis Table II (running median, rho0=7) as a reasonable 4-yr starting point.
INIT = {
    "karnesis": dict(theta=[1.15e-44, 1.56, 1.5e-3, 6.7e-4, 1.9e-3],
                     names=["A", "alpha", "f1", "f2", "fknee"]),
    "robson":   dict(theta=[1.8e-44, 0.138, -221.0, 521.0, 1680.0, 1.13e-3],
                     names=["A", "alpha", "beta", "kappa", "gamma", "fk"]),
}
# Loose physical bounds (A and the frequencies positive; shapes in sane ranges).
BOUNDS = {
    "karnesis": [(1e-48, 1e-40), (0.2, 5.0), (1e-4, 2e-2), (1e-5, 1e-2), (1e-4, 2e-2)],
    "robson":   [(1e-48, 1e-40), (0.05, 1.0), (-1e3, 1e3), (0.0, 3e3), (0.0, 5e3), (1e-4, 1e-2)],
}


# ----------------------------------------------------------------- likelihood
def make_logprob(f, Sdata, instr, channels, model, log_sigma):
    bounds = BOUNDS[model]

    def log_prior(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo <= v <= hi):
                return -np.inf
        return 0.0

    def log_like(theta):
        conf = model_conf(f, theta, model)
        if np.any(~np.isfinite(conf)) or np.any(conf < 0):
            return -np.inf
        ll = 0.0
        for k in channels:
            total = instr[k] + conf
            resid = np.log10(Sdata[k]) - np.log10(total)
            ll += -0.5 * np.sum((resid / log_sigma) ** 2)
        return ll

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_like(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    return log_prob


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="{code}_output_cat_snr{X}.h5 from main_loop")
    ap.add_argument("--code", default=None)
    ap.add_argument("--model", default="karnesis", choices=("karnesis", "robson"))
    ap.add_argument("--channels", default="A,E", help="comma list among A,E (T has ~no confusion)")
    ap.add_argument("--window", type=int, default=2000, help="median window [bins] (config window_size)")
    ap.add_argument("--fmin", type=float, default=5e-5, help="fit band low edge [Hz]")
    ap.add_argument("--fmax", type=float, default=8e-3, help="fit band high edge [Hz]")
    ap.add_argument("--conf-floor", type=float, default=1.3,
                    help="only fit bins where median total > this * instrument (confusion-dominated)")
    ap.add_argument("--log-sigma", type=float, default=0.05, help="log10-PSD scatter for the Gaussian likelihood")
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--nburn", type=int, default=1000)
    ap.add_argument("--out", default="fits", help="output directory for coefficients + plots")
    args = ap.parse_args()

    code = args.code or os.path.basename(args.h5).split("_output_cat")[0]
    channels = tuple(c.strip() for c in args.channels.split(","))
    os.makedirs(args.out, exist_ok=True)

    # --- preprocess: median-smoothed total PSD on the residual ---
    f_full, aet = load_residual_tdi(args.h5)
    instr_full = instrument_aet(f_full)
    f, Smed = median_total_psd(f_full, aet, instr_full, args.window)
    instr = instrument_aet(f)

    # --- fit mask: confusion-dominated band ---
    band = (f >= args.fmin) & (f <= args.fmax)
    conf_dom = Smed["A"] > args.conf_floor * instr["A"]
    mask = band & conf_dom
    if mask.sum() < 50:
        raise SystemExit(f"[{code}] only {mask.sum()} bins in the fit band/confusion region; "
                         f"loosen --fmin/--fmax/--conf-floor.")
    print(f"[{code}] fitting {mask.sum()} bins, f in [{f[mask].min():.2e}, {f[mask].max():.2e}] Hz, "
          f"channels {channels}, model {args.model}")

    fm = f[mask]
    Sd = {k: Smed[k][mask] for k in channels}
    Im = {k: instr[k][mask] for k in channels}

    # --- emcee ---
    try:
        import emcee
    except ImportError:
        raise SystemExit("emcee not installed. In the env:  pip install emcee corner")

    init = INIT[args.model]
    p0 = np.array(init["theta"], dtype=np.float64)
    ndim = len(p0)
    log_prob = make_logprob(fm, Sd, Im, channels, args.model, args.log_sigma)

    rng = np.random.default_rng(0)
    pos = p0 * (1.0 + 1e-3 * rng.standard_normal((args.nwalkers, ndim)))
    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, args.nsteps, progress=True)
    chain = sampler.get_chain(discard=args.nburn, flat=True)

    med = np.median(chain, axis=0)
    lo, hi = np.percentile(chain, [16, 84], axis=0)
    coeffs = {init["names"][i]: dict(median=float(med[i]),
                                     minus=float(med[i] - lo[i]),
                                     plus=float(hi[i] - med[i])) for i in range(ndim)}
    result = dict(code=code, model=args.model, channels=list(channels),
                  fit_band_hz=[float(args.fmin), float(args.fmax)],
                  window=args.window, log_sigma=args.log_sigma,
                  n_bins=int(mask.sum()), coefficients=coeffs)
    out_json = os.path.join(args.out, f"{code}_confusion_fit_{args.model}.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{code}] coefficients -> {out_json}")
    for name in init["names"]:
        c = coeffs[name]
        print(f"    {name:6s} = {c['median']:.4g} (+{c['plus']:.2g}/-{c['minus']:.2g})")

    # --- overlay plot: median data, mean (saved) curve, instrument, fit ---
    _, Ssaved = load_saved_psd(args.h5)
    conf_fit = model_conf(f, med, args.model)
    with plt.rc_context({"font.size": 12, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(9, 6))
        kref = channels[0]
        ax.loglog(f, Smed[kref], color="0.6", lw=1, label=f"median total ({kref}, fit input)")
        if kref in Ssaved:
            ax.loglog(f_full, Ssaved[kref], color="orange", lw=0.8, alpha=0.7,
                      label=f"pipeline mean total ({kref})")
        ax.loglog(f, instr[kref], "k--", lw=1.5, label="instrument (TDI1)")
        ax.loglog(f, instr[kref] + conf_fit, "r", lw=2, label=f"fit: instr + S_conf ({args.model})")
        ax.axvspan(args.fmin, args.fmax, color="b", alpha=0.05)
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(max(instr[kref].min() * 0.3, 1e-44), Smed[kref].max() * 3)
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel(r"PSD [1/Hz]")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", linestyle=":", linewidth=1.0)
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=8)
        fig.tight_layout()
        out_png = os.path.join(args.out, f"{code}_confusion_fit_{args.model}.png")
        fig.savefig(out_png, dpi=150)
        print(f"[{code}] overlay -> {out_png}")
        plt.close(fig)

    # --- corner (optional) ---
    try:
        import corner
        fig = corner.corner(chain, labels=init["names"], show_titles=True)
        fig.savefig(os.path.join(args.out, f"{code}_confusion_fit_{args.model}_corner.png"), dpi=120)
        plt.close(fig)
    except ImportError:
        print("    (corner not installed; skipping posterior corner plot)")


if __name__ == "__main__":
    main()
