"""
No-foreground reference SNR counts for the DWD recovery denominator.

Recovery = resolved / reference. The resolved numerator comes from gwg.icloop
(gbgpu + lisatools). For an apples-to-apples denominator we count sources whose
no-confusion SNR exceeds a threshold using the SAME machinery:

  - gbgpu  : gbgpu.run_wave per source, PER-SOURCE sky position/orientation,
             instrument-only PSD (TDI1 lisatools AET1 / TDI2 fomweb AET).
  - legwork: legwork.snr.snr_circ_stationary with confusion_noise=None
             (sky/pol/inclination-AVERAGED response; included as a cross-check
             and to expose the averaging systematic vs gbgpu).

Both are NO-foreground (instrument noise only) -> the resolvability ceiling.

Importable:
    reference_counts(input_cat, tobs, dt, tdi, thresholds, use_gpu,
                     alldwds=None, methods=("gbgpu","legwork")) -> {method:{thr:count}}
    per_source_snr_gbgpu(cat_df, tobs, dt, tdi, use_gpu) -> snr[]
    per_source_snr_legwork(alldwds_df, tobs) -> snr[]      # sky-averaged

CLI (standalone, e.g. on Thorny Flat over the full catalogs):
    python reference_snr.py --input-cat input/.../COSMIC_input_cat.h5 \
        --alldwds data/.../COSMIC_Galaxy_AllDWDs.csv --code COSMIC \
        --tobs 4 --dt 10 --tdi 1 --gpu --thresholds 5,6,7,8,9 --out ref.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from helpers import Constants  # use the SAME year as the pipeline (gen_waveforms/main_loop)

try:
    from tqdm import tqdm as _tqdm
except ImportError:                      # tqdm optional; fall back to a no-op
    def _tqdm(iterable, **kwargs):
        return iterable


def _to_numpy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def _instrument_psd_fn(tdi: int):
    """Return f(freqs2d) -> (S_A, S_E, S_T), instrument-only, floored."""
    if tdi == 1:
        from lisatools.sensitivity import AET1SensitivityMatrix
        import lisatools.detector as lisa_models

        def psd(freqs2d):
            flat = np.maximum(freqs2d.ravel(), 1e-7)
            sm = AET1SensitivityMatrix(flat, model=lisa_models.scirdv1,
                                       return_type="PSD")
            out = [np.asarray(sm[k]).reshape(freqs2d.shape) for k in range(3)]
            for s in out:
                np.maximum(s, 1e-60, out=s)
            return out
        return psd

    if tdi == 2:
        from fomweb.analytic_noise import InstrumentalNoise
        fom = InstrumentalNoise("scird")

        def psd(freqs2d):
            flat = np.maximum(freqs2d.ravel(), 1e-7)
            out = [fom.psd(flat, option=o).reshape(freqs2d.shape)
                   for o in ("A", "E", "T")]
            for s in out:
                np.maximum(s, 1e-60, out=s)
            return out
        return psd

    raise ValueError(f"tdi must be 1 or 2, got {tdi}")


_GBGPU_COLS = ("Amplitude", "Frequency", "FrequencyDerivative", "InitialPhase",
               "Inclination", "Polarization", "EclipticLongitude", "EclipticLatitude")


def _snr_gbgpu_single(cols, tobs, dt, tdi, use_gpu, batch, progress=True):
    """Per-source no-FG SNR for one set of parameter arrays on the CURRENT device (the
    caller selects the GPU, or CPU). cols = arrays in the order of _GBGPU_COLS."""
    from gbgpu.gbgpu import GBGPU
    import lisatools.detector as lisa_models
    # Match the pipeline's construction (gen_waveforms/main_loop): the orbits' GPU setting
    # must match GBGPU's, else orbits.get_pos hits a GPU/CPU array mismatch when use_gpu=True.
    orbits = lisa_models.EqualArmlengthOrbits(use_gpu=use_gpu)
    gb = GBGPU(orbits=orbits, use_gpu=use_gpu)
    psd = _instrument_psd_fn(tdi)
    T_sec = tobs * Constants.yr
    df_bin = 1.0 / T_sec
    amp, f0, fdot, phi0, iota, psi, lam, beta = cols
    n = len(amp)
    snr = np.empty(n, dtype=np.float64)
    loop = range(0, n, batch)
    if progress:
        loop = _tqdm(loop, total=(n + batch - 1) // batch, desc=f"gbgpu SNR ({n:,} src)", unit="batch")
    for lo in loop:
        hi = min(lo + batch, n)
        sl = slice(lo, hi)
        gb.run_wave(amp[sl], f0[sl], fdot[sl], np.zeros(hi - lo),
                    phi0[sl], iota[sl], psi[sl], lam[sl], beta[sl],
                    T=T_sec, dt=dt, oversample=1, tdi2=(tdi == 2))
        A = _to_numpy(gb.A); E = _to_numpy(gb.E); Tc = _to_numpy(gb.T)
        freqs = _to_numpy(gb.freqs)
        S_A, S_E, S_T = psd(freqs)
        snr2 = 4.0 * df_bin * (
            np.sum(np.abs(A) ** 2 / S_A, axis=1)
            + np.sum(np.abs(E) ** 2 / S_E, axis=1)
            + np.sum(np.abs(Tc) ** 2 / S_T, axis=1)
        )
        snr[sl] = np.sqrt(np.maximum(snr2, 0.0))
    return snr


def _snr_gbgpu_chunk_worker(args):
    """Top-level multiprocessing worker: bind to a GPU, compute the chunk's SNR. CUDA is
    touched ONLY here (never in the parent) so the fork-based pool is safe (cf. gen_waveforms)."""
    cols, tobs, dt, tdi, device_id, batch = args
    if device_id is not None:
        import cupy as cp
        cp.cuda.Device(device_id).use()
    return _snr_gbgpu_single(cols, tobs, dt, tdi, use_gpu=(device_id is not None),
                             batch=batch, progress=False)


def per_source_snr_gbgpu(cat_df, tobs, dt, tdi=1, use_gpu=False, batch=10000):
    """Per-source no-FG SNR with gbgpu + instrument-only PSD (actual sky position).

    On a multi-GPU allocation the catalogue is split into one chunk per visible GPU and the
    chunks run in parallel (one process per GPU, mirroring gen_waveforms), then the SNRs are
    concatenated back in catalogue order. Single-GPU / CPU runs use the same per-chunk code on
    one device. The parent does only numpy splitting + getDeviceCount (no CUDA context) so the
    fork pool is safe.
    """
    cols = tuple(cat_df[c].values.astype(np.float64) for c in _GBGPU_COLS)
    n = len(cols[0])
    num_gpus = 0
    if use_gpu:
        import cupy as cp
        num_gpus = cp.cuda.runtime.getDeviceCount()
    if use_gpu and num_gpus > 1:
        import multiprocessing as mp
        idx = np.array_split(np.arange(n), num_gpus)
        tasks = [(tuple(c[ix] for c in cols), tobs, dt, tdi, g, batch)
                 for g, ix in enumerate(idx)]
        print(f"gbgpu SNR: {n:,} sources across {num_gpus} GPUs ({[len(ix) for ix in idx]} per chunk)")
        # MUST use 'spawn', not the default 'fork': the parent has a CUDA context (even just
        # getDeviceCount initializes it) and a CUDA context does not survive fork, so forked
        # workers fail at setDevice with cudaErrorInitializationError. Spawn = fresh processes
        # that initialize CUDA cleanly (this is what gen_waveforms does).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
            parts = pool.map(_snr_gbgpu_chunk_worker, tasks)
        return np.concatenate(parts)
    return _snr_gbgpu_single(cols, tobs, dt, tdi, use_gpu, batch, progress=True)


def per_source_snr_legwork(alldwds_df, tobs, confusion=None, batch=200000):
    """Per-source SNR with LEGWORK (sky/pol/inclination-AVERAGED).

    Reads physical params from a *_Galaxy_AllDWDs.csv frame: mass1, mass2 [Msun],
    RRelkpc [kpc], PSetTodayHours [hr] (orbital period). Uses snr_circ_stationary
    (NOT Source.get_snr) with confusion_noise=`confusion`:
      - confusion=None        -> instrument-only (no-FG); the recovery denominator.
      - confusion="robson19"  -> LEGWORK galactic foreground ON; reproduces the
        *_Galaxy_LISA_DWDs.csv counts (those were built with get_snr's robson19 default).
    """
    import astropy.units as u
    import legwork.snr as lsnr
    import legwork.utils as lutils

    m1 = alldwds_df["mass1"].values.astype(np.float64)
    m2 = alldwds_df["mass2"].values.astype(np.float64)
    dist = alldwds_df["RRelkpc"].values.astype(np.float64)
    period_hr = alldwds_df["PSetTodayHours"].values.astype(np.float64)
    n = len(m1)
    snr = np.empty(n, dtype=np.float64)

    n_batches = (n + batch - 1) // batch
    for lo in _tqdm(range(0, n, batch), total=n_batches,
                    desc=f"legwork SNR [{confusion or 'no-conf'}] ({n:,} src)", unit="batch"):
        hi = min(lo + batch, n)
        sl = slice(lo, hi)
        m_c = lutils.chirp_mass(m1[sl] * u.Msun, m2[sl] * u.Msun)
        f_orb = (1.0 / (period_hr[sl] * u.hr)).to(u.Hz)
        s = lsnr.snr_circ_stationary(
            m_c=m_c, f_orb=f_orb, dist=dist[sl] * u.kpc,
            t_obs=(tobs * Constants.yr) * u.s,   # same T_obs seconds as the pipeline
            instrument="LISA", confusion_noise=confusion,
        )
        snr[sl] = np.asarray(s, dtype=np.float64)
    return snr


def reference_counts(input_cat, tobs, dt, tdi=1, thresholds=(5, 6, 7, 8, 9),
                     use_gpu=False, alldwds=None, methods=("gbgpu", "legwork"),
                     legwork_confusion=None):
    """Return {method: {threshold: count}} of sources above each threshold.

    References are no-foreground (the recovery denominator) EXCEPT the legwork method
    honors `legwork_confusion` (None = no-FG; "robson19" reproduces the CSV LISA_DWDs
    counts). Leave None for the recovery analysis; set "robson19" only to verify the
    CSV-vs-no-FG discrepancy.
    """
    out = {}
    if "gbgpu" in methods:
        cat = pd.read_hdf(input_cat, key="cat")
        snr = per_source_snr_gbgpu(cat, tobs, dt, tdi=tdi, use_gpu=use_gpu)
        out["gbgpu"] = {float(t): int((snr > t).sum()) for t in thresholds}
    if "legwork" in methods:
        if alldwds is None or not os.path.exists(alldwds):
            print(f"WARNING: legwork method skipped (AllDWDs csv not given/found: {alldwds})")
        else:
            try:
                df = pd.read_csv(alldwds)
                snr = per_source_snr_legwork(df, tobs, confusion=legwork_confusion)
                out["legwork"] = {float(t): int((snr > t).sum()) for t in thresholds}
            except ImportError as e:
                print(f"WARNING: legwork not installed; skipping legwork cross-check ({e}). "
                      f"The gbgpu reference is unaffected.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-cat", required=True, help="*_input_cat.h5 (gbgpu method)")
    ap.add_argument("--alldwds", default=None, help="*_Galaxy_AllDWDs.csv (legwork method)")
    ap.add_argument("--code", default=None)
    ap.add_argument("--tobs", type=float, default=4.0, help="observation time [yr]")
    ap.add_argument("--dt", type=float, default=10.0, help="cadence [s]")
    ap.add_argument("--tdi", type=int, default=1, choices=(1, 2))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--thresholds", default="5,6,7,8,9")
    ap.add_argument("--methods", default="gbgpu,legwork")
    ap.add_argument("--legwork-confusion", default="none",
                    help="legwork confusion model: 'none' (no-FG, default) or 'robson19' "
                         "(reproduces the CSV LISA_DWDs counts — verification only)")
    ap.add_argument("--out", default=None, help="CSV to append results to")
    args = ap.parse_args()

    code = args.code or os.path.basename(args.input_cat).replace("_input_cat.h5", "")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    methods = tuple(m.strip() for m in args.methods.split(","))
    legwork_confusion = (None if args.legwork_confusion.lower() in ("none", "", "null")
                         else args.legwork_confusion)

    print(f"[{code}] T_obs={args.tobs} yr, dt={args.dt} s, TDI{args.tdi}, "
          f"gpu={args.gpu}, methods={methods}, legwork_confusion={legwork_confusion}")
    counts = reference_counts(args.input_cat, args.tobs, args.dt, tdi=args.tdi,
                              thresholds=thresholds, use_gpu=args.gpu,
                              alldwds=args.alldwds, methods=methods,
                              legwork_confusion=legwork_confusion)

    rows = []
    for method, by_thr in counts.items():
        print(f"  {method}:")
        for thr in thresholds:
            print(f"    SNR>{thr:g}: {by_thr[float(thr)]:,}")
            rows.append({"code": code, "method": method, "tdi": args.tdi,
                         "tobs_yr": args.tobs, "dt_s": args.dt,
                         "snr_threshold": thr, "ref_count": by_thr[float(thr)]})
    if args.out:
        out_df = pd.DataFrame(rows)
        if os.path.exists(args.out):
            out_df = pd.concat([pd.read_csv(args.out), out_df], ignore_index=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
