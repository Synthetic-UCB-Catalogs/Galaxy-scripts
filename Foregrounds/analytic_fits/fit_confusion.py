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
    python analytic_fits/fit_confusion.py --dir output/<...>/fiducial --code COSMIC --snr 7 \
        --out analytic_fits/fits
    # the h5 name is built as {code}_output_cat_snr{snr}.h5; or point directly with --h5.
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


# ------------------------------------------------------------- coarse-graining
def coarse_grain(f, S, fmin, fmax, n_per_decade=250, sigma_floor=0.05):
    """Log-bin the in-band curve into ~uncorrelated points (Karnesis-style).

    The median curve is window-correlated (~2000-bin filter), so fitting every bin
    as independent overcounts information and shrinks the error bars. We bin to
    ~n_per_decade log-spaced bins and fit one point per bin. Returns (fb, {k: Sb},
    {k: sigma}): geometric bin centres, the per-bin mean of log10(S) back in linear,
    and the std of log10(S) within the bin (the local scatter) floored at sigma_floor.
    Bins with <2 samples are dropped.
    """
    if f.size == 0 or fmax <= fmin:
        empty = np.array([])
        return empty, {k: empty for k in S}, {k: empty for k in S}
    nbins = max(int(np.ceil(n_per_decade * np.log10(fmax / fmin))), 4)
    edges = np.logspace(np.log10(fmin), np.log10(fmax), nbins + 1)
    which = np.digitize(f, edges)
    fb, Sb, sig = [], {k: [] for k in S}, {k: [] for k in S}
    for b in range(1, nbins + 1):
        sel = which == b
        if sel.sum() < 2:
            continue
        fb.append(10 ** np.mean(np.log10(f[sel])))
        for k in S:
            logS = np.log10(S[k][sel])
            Sb[k].append(10 ** np.mean(logS))
            sig[k].append(max(float(np.std(logS)), sigma_floor))
    fb = np.asarray(fb)
    return fb, {k: np.asarray(v) for k, v in Sb.items()}, {k: np.asarray(v) for k, v in sig.items()}


# ----------------------------------------------------------------- likelihood
# The likelihood payload lives in a module global so emcee's multiprocessing pool
# workers read the (large) data arrays without re-pickling them on every call:
# only the tiny theta vector crosses the process boundary per evaluation. A Pool
# initializer sets _FIT in each worker (portable across fork/spawn).
_FIT = {}


def _init_worker(payload):
    global _FIT
    _FIT = payload


def _log_prob(theta):
    """Gaussian-in-log10(PSD) posterior for S_total = instrument + S_conf(theta)."""
    F = _FIT
    for v, (lo, hi) in zip(theta, F["bounds"]):
        if not (lo <= v <= hi):
            return -np.inf
    conf = model_conf(F["f"], theta, F["model"])
    if not np.all(np.isfinite(conf)) or np.any(conf < 0):
        return -np.inf
    ll = 0.0
    for k in F["channels"]:
        resid = F["logSd"][k] - np.log10(F["Im"][k] + conf)
        ll += -0.5 * np.sum((resid / F["logSig"][k]) ** 2)
    return ll if np.isfinite(ll) else -np.inf


# ------------------------------------------------------------------- plotting
def plot_curves(out_png, f, Smed, fS, Ssaved, instr, channel, fmin, fmax,
                model_curve=None, model_label="fit", binned=None):
    """Overlay: median total (fit input), pipeline mean total, instrument, the
    coarse-grained fit points, and an optional fitted model."""
    xlo, xhi = 1e-4, 2e-2
    with plt.rc_context({"font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.loglog(f, Smed[channel], color="0.6", lw=1.0, label=f"median total ({channel})")
        if Ssaved is not None and channel in Ssaved:
            ax.loglog(fS, Ssaved[channel], color="orange", lw=0.7, alpha=0.7,
                      label=f"pipeline mean total ({channel})")
        ax.loglog(f, instr[channel], "k--", lw=1.5, label="instrument (TDI1)")
        if binned is not None:
            fb, Sb, sg = binned
            ax.errorbar(fb, Sb, yerr=[Sb - Sb * 10 ** (-sg), Sb * 10 ** sg - Sb],
                        fmt=".", ms=4, color="C0", lw=0.7, capsize=2, zorder=5,
                        label="coarse-grained (fit pts)")
        if model_curve is not None:
            ax.loglog(f, model_curve, "r", lw=2, zorder=6, label=model_label)
        ax.axvspan(fmin, fmax, color="b", alpha=0.06)
        ax.set_xlim(xlo, xhi)
        view = (f >= xlo) & (f <= xhi)               # scale y to the in-view curve, not the
        ymax = float(np.nanmax(Smed[channel][view]))  # low-f instrument blow-up out of frame
        ymin = max(float(np.nanmin(instr[channel][view])) * 0.5, 1e-44)
        ax.set_ylim(ymin, ymax * 3)
        ax.set_xlabel(r"Frequency [Hz]")
        ax.set_ylabel(r"PSD [1/Hz]")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", linestyle=":", linewidth=1.0)
        ax.tick_params("both", length=3, width=0.5, which="both", direction="in", pad=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    print(f"    overlay -> {out_png}")


def _load_fit_config(path):
    """Read a YAML of defaults for the CLI (returns {} if the file is absent)."""
    if path and os.path.exists(path):
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _default_nproc():
    """CPUs actually available to this process: SLURM allocation, else cpu affinity,
    else total cores. (Pool()'s own default is os.cpu_count() = all NODE cores, which
    oversubscribes a shared SLURM node; SLURM sets a cpuset that affinity respects.)"""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# ----------------------------------------------------------------------- main
def main():
    # Two-stage parse: read --config first so its entries become argparse defaults
    # (any explicit CLI flag still overrides the config file).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_config.yaml"),
                     help="YAML of MCMC/fit defaults next to this script (CLI flags override it)")
    known, _ = pre.parse_known_args()
    cfg = _load_fit_config(known.config)
    chan_default = cfg.get("channels", ["A", "E"])
    chan_default = ",".join(chan_default) if isinstance(chan_default, list) else str(chan_default)

    ap = argparse.ArgumentParser(parents=[pre], description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="output dir holding {code}_output_cat_snr{snr}.h5 "
                         "(e.g. .../initial_condition_variations/fiducial)")
    ap.add_argument("--code", default=None,
                    help="stellar code (e.g. COSMIC); with --dir/--snr it builds the h5 filename")
    ap.add_argument("--snr", type=float, default=7,
                    help="SNR cutoff identifying the run; file suffix snr{snr:g} (default 7)")
    ap.add_argument("--h5", default=None, help="explicit h5 path; overrides --dir/--code/--snr")
    ap.add_argument("--model", default=cfg.get("model", "karnesis"), choices=("karnesis", "robson"))
    ap.add_argument("--channels", default=chan_default, help="comma list among A,E (T has ~no confusion)")
    ap.add_argument("--window", type=int, default=cfg.get("window", 2000),
                    help="median window [bins] (config window_size)")
    ap.add_argument("--fmin", type=float, default=cfg.get("fmin", None),
                    help="fit band low edge [Hz]; default = inferred from the curve (lowest valid bin)")
    ap.add_argument("--fmax", type=float, default=cfg.get("fmax", None),
                    help="fit band high edge [Hz]; default = inferred from the curve (highest valid bin)")
    ap.add_argument("--fmax-crop", type=float, default=cfg.get("fmax_crop", 2e-2),
                    help="crop residual to f<=this before median-filtering (speed only; keep >> the knee)")
    ap.add_argument("--conf-floor", type=float, default=cfg.get("conf_floor", 1.3),
                    help="restrict the fit to bins where median total > this * instrument (confusion-dominated)")
    ap.add_argument("--n-per-decade", type=int, default=cfg.get("n_per_decade", 250),
                    help="log-bins per frequency decade for coarse-graining the fit curve")
    ap.add_argument("--log-sigma", type=float, default=cfg.get("log_sigma", 0.05),
                    help="floor on the per-bin log10-PSD scatter used as the Gaussian sigma")
    ap.add_argument("--nwalkers", type=int, default=cfg.get("nwalkers", 32))
    ap.add_argument("--nsteps", type=int, default=cfg.get("nsteps", 5000))
    ap.add_argument("--nburn", type=int, default=cfg.get("nburn", 1500))
    ap.add_argument("--nproc", type=int, default=cfg.get("nproc", None),
                    help="emcee CPU workers; default auto-detect (SLURM allocation / available cores); 1 = serial")
    ap.add_argument("--preprocess-only", action="store_true",
                    help="build + plot the median curve vs the pipeline mean curve, then exit (no emcee)")
    ap.add_argument("--out", default=cfg.get("out", "fits"), help="output directory for coefficients + plots")
    args = ap.parse_args()

    if args.h5:
        h5 = args.h5
        code = args.code or os.path.basename(h5).split("_output_cat")[0]
    elif args.dir and args.code:
        code = args.code
        h5 = os.path.join(args.dir, f"{code}_output_cat_snr{args.snr:g}.h5")
    else:
        ap.error("provide --dir and --code (or an explicit --h5)")
    if not os.path.exists(h5):
        ap.error(f"h5 not found: {h5}")
    channels = tuple(c.strip() for c in args.channels.split(","))
    os.makedirs(args.out, exist_ok=True)

    # --- preprocess: median-smoothed total PSD on the residual ---
    f_full, aet = load_residual_tdi(h5)
    instr_full = instrument_aet(f_full)
    f, Smed = median_total_psd(f_full, aet, instr_full, args.window, fmax_crop=args.fmax_crop)
    instr = instrument_aet(f)
    fS, Ssaved = load_saved_psd(h5)        # the pipeline's mean-smoothed curve, for comparison
    ref = channels[0]

    # --- fit band: inferred from the curve itself (its frequency extent minus the
    # median-filter edge bins, restricted to the confusion-dominated region, which sets
    # the high edge at the knee). --fmin/--fmax override; --conf-floor sets "dominated". ---
    edge = max(args.window // 2, 1)
    valid = np.zeros(f.size, dtype=bool)
    valid[edge:f.size - edge] = True
    conf_dom = Smed[ref] > args.conf_floor * instr[ref]
    fmin = args.fmin if args.fmin is not None else float(f[valid].min())
    fmax = args.fmax if args.fmax is not None else float(f[valid].max())
    mask = valid & conf_dom & (f >= fmin) & (f <= fmax)
    if mask.any():
        fmin, fmax = float(f[mask].min()), float(f[mask].max())   # tighten to the actual fitted bins
    print(f"[{code}] median curve: {f.size} bins up to {f.max():.2e} Hz; "
          f"confusion-dominated fit band [{fmin:.2e}, {fmax:.2e}] Hz ({mask.sum()} bins), "
          f"channels {channels}, model {args.model}")

    # --- coarse-grain the confusion-dominated band to ~uncorrelated log points ---
    fb, Sb, sigb = coarse_grain(f[mask], {k: Smed[k][mask] for k in channels}, fmin, fmax,
                                n_per_decade=args.n_per_decade, sigma_floor=args.log_sigma)
    Imb = instrument_aet(fb) if fb.size else {k: np.array([]) for k in channels}
    binned = (fb, Sb[ref], sigb[ref]) if fb.size else None
    print(f"[{code}] coarse-grained to {fb.size} log-bins (~{args.n_per_decade}/decade) "
          f"from {int(mask.sum())} median bins")

    # --- preprocessing-only diagnostic: curves + coarse-grained points, then exit ---
    if args.preprocess_only:
        out_png = os.path.join(args.out, f"{code}_confusion_preprocess.png")
        plot_curves(out_png, f, Smed, fS, Ssaved, instr, ref, fmin, fmax, binned=binned)
        print(f"[{code}] preprocess-only: wrote median-vs-mean comparison, skipping fit.")
        return

    if fb.size < 10:
        raise SystemExit(f"[{code}] only {fb.size} coarse bins; loosen the band/--conf-floor "
                         f"or lower --n-per-decade.")

    # --- emcee ---
    try:
        import emcee
    except ImportError:
        raise SystemExit("emcee not installed. In the env:  conda install -c conda-forge emcee corner")
    import multiprocessing as mp

    init = INIT[args.model]
    p0 = np.array(init["theta"], dtype=np.float64)
    ndim = len(p0)
    payload = dict(f=fb, Im=Imb, logSd={k: np.log10(Sb[k]) for k in channels},
                   logSig={k: sigb[k] for k in channels},
                   channels=channels, model=args.model, bounds=BOUNDS[args.model])

    rng = np.random.default_rng(0)
    pos = p0 * (1.0 + 1e-3 * rng.standard_normal((args.nwalkers, ndim)))

    nproc = args.nproc if args.nproc is not None else _default_nproc()
    pool = None
    if nproc and nproc > 1:
        pool = mp.Pool(nproc, initializer=_init_worker, initargs=(payload,))
        print(f"[{code}] emcee on {nproc} CPU workers")
    else:
        _init_worker(payload)        # serial: set the payload in this process
    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, _log_prob, pool=pool)
    sampler.run_mcmc(pos, args.nsteps, progress=True)
    if pool is not None:
        pool.close()
        pool.join()

    print(f"[{code}] mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.2f}")
    chain = sampler.get_chain(discard=args.nburn, flat=True)

    med = np.median(chain, axis=0)
    lo, hi = np.percentile(chain, [16, 84], axis=0)
    coeffs = {init["names"][i]: dict(median=float(med[i]),
                                     minus=float(med[i] - lo[i]),
                                     plus=float(hi[i] - med[i])) for i in range(ndim)}
    result = dict(code=code, model=args.model, channels=list(channels),
                  fit_band_hz=[fmin, fmax], n_per_decade=args.n_per_decade,
                  n_points=int(fb.size), n_median_bins=int(mask.sum()),
                  window=args.window, sigma_floor=args.log_sigma, coefficients=coeffs)
    out_json = os.path.join(args.out, f"{code}_confusion_fit_{args.model}.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{code}] coefficients -> {out_json}")
    for name in init["names"]:
        c = coeffs[name]
        print(f"    {name:6s} = {c['median']:.4g} (+{c['plus']:.2g}/-{c['minus']:.2g})")

    # --- overlay plot: median data, pipeline mean curve, instrument, fit ---
    conf_fit = model_conf(f, med, args.model)
    out_png = os.path.join(args.out, f"{code}_confusion_fit_{args.model}.png")
    plot_curves(out_png, f, Smed, fS, Ssaved, instr, ref, fmin, fmax,
                model_curve=instr[ref] + conf_fit, model_label=f"fit: instr + S_conf ({args.model})",
                binned=binned)

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
