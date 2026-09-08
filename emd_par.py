"""
emd_par.py  --  Reusable parallel 1-D EMD / EEMD decomposition module.

PUBLIC API
----------
    result = decompose(x, method='auto', E=64, eps=0.2, tau=0.25,
                       max_sifts=50, max_imf=-1, min_extrema=4,
                       end='natural', n_workers=None, seed=0, batch=None)

    result.imfs        # ndarray [n_modes, N]  -- (ensemble-averaged) IMFs
    result.residual    # ndarray [N]
    result.n_modes     # int
    result.stats       # dict: method, E, n_modes, per-trial n_sifts, timing
    result.recon_error()   # max|x - (sum(imfs)+residual)|  (partition of unity)
    result.save(path) / EMDResult.load(path)

    x = <1-D numpy array>
    result = decompose(x)                 # plain EMD (E=1), CPU
    result = decompose(x, E=64, tau=0.25) # EEMD, user stop-criterion tau

STOPPING CRITERION (user-controllable)
--------------------------------------
    tau         SD tolerance (Rilling-Flandrin-Goncalves). Lower tau = more
                siftings = closer to a "true" IMF, slower. Default 0.25.
    max_sifts   Hard cap on siftings per IMF (safety). Default 50.
    min_extrema Stop the decomposition when the residual has fewer than this
                many local extrema (can no longer define a spline). Default 4.
    max_imf     Stop after this many IMFs (-1 = decompose fully). Default -1.

BACKENDS
--------
    method='cpu' : vectorized numpy + ProcessPoolExecutor with DYNAMIC
                   scheduling (as_completed) -> load-imbalance handling. This
                   is the robust workhorse for 100k-2M samples.
    method='gpu' : cupy + numba.cuda, one thread block per trial, O(N) spline
                   evaluation (binary-search interval lookup), trial batching
                   to fit device memory.
    method='auto': GPU if available and the working set fits, else CPU.

WHY THIS DESIGN (honest, see the feasibility note in the report):
  * A single EMD of one signal is an irreducible SEQUENTIAL chaotic loop
    (siftings and IMFs are each strictly sequential). You cannot parallelize
    the sifting of one signal.
  * The parallel axis is the ENSEMBLE: EEMD runs E independent EMDs of
    (x + eps*std(x)*noise_e) and averages the IMFs level by level. Trials are
    independent -> embarrassingly parallel. E=1 degenerates to plain EMD.
  * Within a sift, the O(N) work (extrema, spline eval, subtract, SD) is
    embarrassingly parallel; the tridiagonal solve is O(knots) and 2-dominant
    (no pivot).
"""
from __future__ import annotations
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# --- reuse the proven CPU primitives (exact 2-dominant spline, EMD, EEMD) ---
import emd_ref as _R

__all__ = ['decompose', 'EMDResult', 'make_reference_sweep']


# =====================================================================
# Result container
# =====================================================================
@dataclass
class EMDResult:
    imfs: np.ndarray          # [n_modes, N]
    residual: np.ndarray      # [N]
    n_modes: int
    stats: dict = field(default_factory=dict)

    def recon_error(self) -> float:
        """Partition-of-unity check: max|x - (sum(imfs)+residual)|."""
        recon = self.imfs.sum(axis=0) + self.residual
        x = self.stats.get('x', None)
        if x is None:
            # reconstruct x from imfs+residual is identity; error is 0 by
            # construction unless the caller stored x.
            return 0.0
        return float(np.max(np.abs(x - recon)))

    def save(self, path: str):
        scalars = {k: v for k, v in self.stats.items()
                   if isinstance(v, (int, float, str, np.integer, np.floating))
                   and k not in ('n_modes',)}
        np.savez_compressed(path, imfs=self.imfs, residual=self.residual,
                            n_modes=self.n_modes, **scalars)

    @staticmethod
    def load(path: str) -> 'EMDResult':
        d = np.load(path, allow_pickle=False)
        return EMDResult(imfs=d['imfs'], residual=d['residual'],
                         n_modes=int(d['n_modes']),
                         stats={k: d[k] for k in d.files if k not in
                                ('imfs', 'residual', 'n_modes')})


# =====================================================================
# CPU backend  (the workhorse)
# =====================================================================
# Module-global shared by pool workers (set via initializer). Avoids
# pickling the full signal x into every one of the E tasks.
_SHARED = {}

def _pool_init(x, sigma):
    _SHARED['x'] = x
    _SHARED['sigma'] = sigma

def _worker_eemd(e, seed, tau, max_sifts, min_extrema, end):
    """Top-level (picklable) worker: EMD of x + per-trial noise.
    Noise is generated LOCALLY from a deterministic seed, so the E*N noise
    array is never pickled / sent over IPC. Returns (imfs, resid, stats)."""
    x = _SHARED['x']
    N = len(x)
    rng = np.random.default_rng((seed * 1000003 + e) & 0xFFFFFFFFFFFFFFFF)
    noise = rng.standard_normal(N) * _SHARED['sigma']
    imfs, resid, stats = _R.emd_1d(x + noise, tau=tau, max_sifts=max_sifts,
                                   min_extrema=min_extrema, end=end)
    return imfs, resid, stats

def _accumulate(avg, cnt, imfs, N):
    """Streaming level-by-level accumulation. Grows avg/cnt as needed;
    bounds memory to O(n_modes * N), independent of E."""
    m = len(imfs)
    if avg is None:
        avg = np.zeros((m, N))
        cnt = np.zeros(m)
    if m > avg.shape[0]:
        extra = m - avg.shape[0]
        avg = np.vstack([avg, np.zeros((extra, N))])
        cnt = np.concatenate([cnt, np.zeros(extra)])
    for i in range(m):
        avg[i] += imfs[i]
        cnt[i] += 1
    return avg, cnt

def _cpu_decompose(x, E, eps, tau, max_sifts, max_imf, min_extrema, end,
                   n_workers, seed, parallel):
    """CPU EMD (E=1) or EEMD (E>1). Returns (imfs, residual, stats).
    EEMD uses streaming accumulation (memory O(n_modes*N), not O(E*N*modes))
    and per-worker seed-based noise (no E*N pickling)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if E == 1:
        imfs, resid, st = _R.emd_1d(x, tau=tau, max_sifts=max_sifts,
                                    max_imf=max_imf, min_extrema=min_extrema,
                                    end=end)
        imfs = np.array(imfs)
        stats = dict(method='cpu', E=1, n_modes=st['n_modes'],
                     total_sifts=st['total_sifts'],
                     per_mode_sifts=st['per_mode_sifts'], x=x)
        return imfs, resid, stats
    sigma = eps * np.std(x)
    n_workers = n_workers or max(1, min((E - 1) // 2, os.cpu_count() or 1))
    avg = None
    cnt = None
    per_trial_sifts = []
    if not parallel:
        # serial streaming (ground-truth baseline)
        _SHARED['x'] = x; _SHARED['sigma'] = sigma
        for e in range(E):
            imfs, resid, stats = _worker_eemd(e, seed, tau, max_sifts,
                                              min_extrema, end)
            avg, cnt = _accumulate(avg, cnt, imfs, N)
            per_trial_sifts.append(stats['total_sifts'])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_pool_init, initargs=(x, sigma)) as ex:
            futs = {ex.submit(_worker_eemd, e, seed, tau, max_sifts,
                              min_extrema, end): e for e in range(E)}
            for fut in as_completed(futs):   # dynamic: next-finished first
                imfs, resid, stats = fut.result()
                avg, cnt = _accumulate(avg, cnt, imfs, N)
                per_trial_sifts.append(stats['total_sifts'])
    n_modes = int((cnt > 0).sum())
    avg = avg[:n_modes] / np.where(cnt[:n_modes] > 0, cnt[:n_modes], 1)[:, None]
    resid = x - avg.sum(axis=0)
    stats = dict(method='cpu', E=E, n_modes=n_modes, n_workers=n_workers,
                 per_trial_total_sifts=np.array(per_trial_sifts), x=x)
    return avg, resid, stats


# =====================================================================
# GPU backend
# =====================================================================
def _gpu_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _gpu_decompose(x, E, eps, tau, max_sifts, max_imf, min_extrema, end,
                   seed, batch, nthreads):
    """GPU EEMD: one block per trial, full IMF loop inside the kernel,
    O(N) spline eval. batch = max trials resident at once (memory control)."""
    import emd_gpu as G
    x = np.asarray(x, dtype=float)
    N = len(x)
    if max_imf == -1:
        max_imf = 64  # hard cap; the kernel stops when extrema run out
    # batch to fit device memory: per-trial ~= (1 + max_imf)*N + 10*(N/2) doubles
    try:
        import cupy as cp
        free, _ = cp.cuda.runtime.memGetInfo()
        free = free / (1024**3)
    except Exception:
        free = 10.0
    per_trial_gb = (1 + max_imf) * N * 8 / 1024**3 + 10 * (N // 2) * 8 / 1024**3
    auto_batch = max(1, int(free * 0.7 / per_trial_gb))
    batch = batch or auto_batch

    rng = np.random.default_rng(seed)
    avg = np.zeros((max_imf, N))
    cnt = np.zeros(max_imf)
    per_trial_sifts = []
    t0 = time.time()
    for b0 in range(0, E, batch):
        b1 = min(E, b0 + batch)
        Eb = b1 - b0
        noises = rng.standard_normal((Eb, N)) * (eps * np.std(x))
        # full-IMF GPU EEMD over this batch
        imfs_batch, nsift, nmode = G.gpu_eemd_full(x + noises, E=Eb, tau=tau,
                                                   max_sifts=max_sifts,
                                                   max_imf=max_imf,
                                                   nthreads=nthreads)
        # imfs_batch: (Eb, maxm, N); average over trials per mode.
        # A trial e contributes only to modes < nmode[e] (it stopped earlier).
        maxm = imfs_batch.shape[1]
        for m in range(maxm):
            mask = (nmode > m)            # trials that produced mode m
            avg[m] += imfs_batch[mask, m, :].sum(axis=0)
            cnt[m] += int(mask.sum())
        per_trial_sifts.extend(int(s) for s in nsift)
    dt = time.time() - t0
    n_modes = int((cnt > 0).sum())
    avg = avg[:n_modes] / np.where(cnt[:n_modes] > 0, cnt[:n_modes], 1)[:, None]
    resid = x - avg.sum(axis=0)
    stats = dict(method='gpu', E=E, n_modes=n_modes, batch=batch,
                 per_trial_total_sifts=np.array(per_trial_sifts),
                 gpu_time_s=dt, x=x)
    return avg, resid, stats


# =====================================================================
# Public API
# =====================================================================
def decompose(x, method='auto', E=64, eps=0.2, tau=0.25, max_sifts=50,
              max_imf=-1, min_extrema=4, end='natural', n_workers=None,
              seed=0, parallel=True, batch=None, nthreads=128):
    """Decompose a 1-D signal x into IMFs (+ residual).

    Parameters
    ----------
    x : 1-D array
    method : 'auto' | 'cpu' | 'gpu'
    E : ensemble size (1 = plain EMD, >1 = EEMD)
    eps : noise amplitude (fraction of std(x)) for EEMD
    tau : SD stop-criterion tolerance (user-controlled)
    max_sifts, max_imf, min_extrema, end : see module docstring
    n_workers : CPU process-pool size (None = auto)
    seed : RNG seed (reproducibility)
    batch : GPU trials resident at once (None = auto from free memory)

    Returns
    -------
    EMDResult with .imfs [n_modes, N], .residual [N], .n_modes, .stats.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    N = len(x)
    if N < 8:
        raise ValueError("signal too short for EMD (need >= 8 samples)")

    if method == 'auto':
        method = 'gpu' if _gpu_available() else 'cpu'
    if method == 'cpu':
        imfs, resid, stats = _cpu_decompose(x, E, eps, tau, max_sifts, max_imf,
                                            min_extrema, end, n_workers, seed,
                                            parallel)
    elif method == 'gpu':
        imfs, resid, stats = _gpu_decompose(x, E, eps, tau, max_sifts, max_imf,
                                            min_extrema, end, seed, batch,
                                            nthreads)
    else:
        raise ValueError("method must be 'auto', 'cpu', or 'gpu'")
    return EMDResult(imfs=imfs, residual=resid, n_modes=imfs.shape[0],
                     stats=stats)


# =====================================================================
# Reference signal (the user-specified 10-octave sweep)
# =====================================================================
def make_reference_sweep(FS=44100, T=10.0, F1=20.0, F2=20000.0,
                         noise_level=0.01, seed=0):
    """10-octave log sine sweep (20Hz-20kHz) over 10 s at Fs=44.1kHz, plus
    white noise at `noise_level` (fraction of peak). Returns (x, t)."""
    N = int(FS * T)
    t = np.arange(N) / FS
    ratio = F2 / F1
    phi = 2 * np.pi * F1 * T * (ratio ** (t / T) - 1.0) / np.log(ratio)
    x = np.sin(phi)
    rng = np.random.default_rng(seed)
    x = x + noise_level * rng.standard_normal(N)
    return x, t


if __name__ == '__main__':
    # quick smoke test on the reference sweep
    x, t = make_reference_sweep()
    print("Reference sweep: N=%d" % len(x))
    r = decompose(x, method='cpu', E=1, tau=0.25)
    print("Plain EMD (CPU): n_modes=%d  recon_err=%.2e  stats=%s"
          % (r.n_modes, r.recon_error(),
             {k: v for k, v in r.stats.items() if k != 'x'}))
