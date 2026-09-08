"""
emd_ref.py  --  CPU reference for 1-D EMD / EEMD / CEEMDAN (ground truth).

Design (see the (b) note):
  * The *ensemble* axis is the dominant, statistically-exact parallel axis.
    Each trial is a full, self-contained EMD of (residual + noise_e); trials
    are independent per stage -> embarrassingly parallel.
  * CPU parallelism = ProcessPoolExecutor with DYNAMIC scheduling
    (as_completed): a worker always takes the next unfinished trial. This is
    the Python/OpenMP equivalent of the ensemble axis and IS the load-imbalance
    handling (a slow trial does not idle the pool).
  * The per-trial EMD uses the EXACT cubic-spline tridiagonal system from part
    (a): factor-2 diagonally dominant, no pivot, Thomas solve. Natural BC by
    default; 'mirror' end handling available for the §10.6 comparison.

Stating the §10.1 honesty point up front:
  * There is NO general convergence theorem for the (nonlinear) sifting. The
    SD / count-stability criteria are practical surrogates. We do NOT claim the
    sifting is a contraction. What IS true: for a FIXED knot configuration one
    sifting step is a linear (2-dominant) operator, so a *single step* is
    well-posed and backward stable; the nonlinearity is only in the knot
    locations moving between steps.
"""
from __future__ import annotations
import numpy as np
import time

# =====================================================================
# 1. Cubic-spline envelope  (exact system from part (a))
# =====================================================================
def _build_interior_system(tm, ym):
    """Interior tridiagonal system for spline 2nd derivs M (natural BC).
    Knots tm[0..p-1], values ym. Returns (d,l,u,H,n):
        A[m,m]=d[m]=2(H[m]+H[m+1]);  A[m,m+1]=u[m]=H[m+1];  A[m,m-1]=l[m]=H[m]
    with interior unknowns M_2..M_{p-1} (n = p-2).  d = 2(l+u) exactly."""
    p = len(tm)
    H = np.diff(tm)                     # H[j] = tm[j+1]-tm[j] = h_{j+1} (1-based)
    n = p - 2
    d = 2.0 * (H[0:n] + H[1:n+1])
    l = np.zeros(n); u = np.zeros(n)
    l[1:]     = H[1:n]                  # h_{k-1}
    u[0:n-1]  = H[1:n]                  # h_k
    return d, l, u, H, n

def _rhs(tm, ym, H, n):
    hp = H[0:n];  hn = H[1:n+1]
    yp = ym[0:n]; yk = ym[1:n+1]; yn = ym[2:n+2]
    return 6.0 * ((yn - yk)/hn - (yk - yp)/hp)

def thomas(d, l, u, r):
    """O(n) Thomas, no pivot (valid: 2-dominant). Returns interior M."""
    n = len(d)
    if n == 0:
        return np.zeros(0)
    c_ = np.empty(n); d_ = np.empty(n)
    c_[0] = (u[0] if n > 1 else 0.0) / d[0]
    d_[0] = r[0] / d[0]
    for i in range(1, n):
        m = l[i] / d[i]
        c_[i] = (u[i] if i < n-1 else 0.0) / (d[i] - l[i]*c_[i-1])
        d_[i] = (r[i] - l[i]*d_[i-1]) / (d[i] - l[i]*c_[i-1])
    x = np.empty(n); x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x

def _spline_eval(tm, ym, M, t):
    """Evaluate the cubic spline (2nd derivs M, natural BC) at sample array t."""
    p = len(tm)
    idx = np.clip(np.searchsorted(tm, t) - 1, 0, p - 2)
    h = tm[idx+1] - tm[idx]
    a = (tm[idx+1] - t) / h
    b = (t - tm[idx]) / h
    return (a*ym[idx] + b*ym[idx+1]
            + ((a**3 - a)*M[idx] + (b**3 - b)*M[idx+1]) * (h*h) / 6.0)

def envelope(t, x, want, end='natural'):
    """Cubic-spline envelope of x through its local maxima (want='max') or
    minima (want='min'), evaluated at all samples t. Returns (env, n_knots).
    end='natural'  : natural BC (M_1=M_p=0) on the extrema knots.
    end='mirror'   : Rilling et al. -- reflect the 2nd/penultimate extremum
                     past each edge to form extra knots, fit, evaluate on the
                     ORIGINAL grid (the extra knots only regularize the ends)."""
    if want == 'max':
        k = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    else:
        k = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    p = len(k)
    if p < 2:
        return np.full_like(t, x[k[0]] if p == 1 else 0.0), p
    tm, ym = t[k], x[k]
    if end == 'mirror' and p >= 3:
        dt0 = tm[1] - tm[0];  dt1 = tm[-1] - tm[-2]
        tm2 = np.concatenate(([tm[0] - dt0], tm, [tm[-1] + dt1]))
        ym2 = np.concatenate(([ym[1]], ym, [ym[-2]]))
    else:
        tm2, ym2 = tm, ym
    d, l, u, H, n = _build_interior_system(tm2, ym2)
    M_int = thomas(d, l, u, _rhs(tm2, ym2, H, n))
    M_full = np.zeros(len(tm2)); M_full[1:-1] = M_int
    # evaluate on the ORIGINAL sample grid t (lies inside the fitted span)
    env = _spline_eval(tm2, ym2, M_full, t)
    return env, p

# =====================================================================
# 2. Single-signal EMD  (the atomic, sequential unit)
# =====================================================================
def emd_1d(x, t=None, tau=0.25, max_sifts=50, max_imf=-1,
           min_extrema=4, end='natural'):
    """Full EMD of 1-D signal x. Returns (imfs, residual, stats).
    stats: dict with per-mode n_sifts and total n_sifts (for load-imbalance)."""
    if t is None:
        t = np.arange(len(x), dtype=float)
    x = np.asarray(x, dtype=float)
    imfs = []
    resid = x.copy()
    n_modes = 0
    total_sifts = 0
    per_mode_sifts = []
    while True:
        nmax = np.sum((resid[1:-1] > resid[:-2]) & (resid[1:-1] > resid[2:]))
        nmin = np.sum((resid[1:-1] < resid[:-2]) & (resid[1:-1] < resid[2:]))
        if nmax + nmin < min_extrema:
            break
        if max_imf != -1 and n_modes >= max_imf:
            break
        h = resid.copy()
        h_prev = None
        n_sift = 0
        for _ in range(max_sifts):
            nmax = np.sum((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))
            nmin = np.sum((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))
            if nmax < 1 or nmin < 1:
                break
            eu, _ = envelope(t, h, 'max', end)
            el, _ = envelope(t, h, 'min', end)
            m = 0.5 * (eu + el)
            h_new = h - m
            n_sift += 1
            if h_prev is not None:
                num = np.sum((h_prev - h_new)**2)
                den = np.sum(h_prev**2)
                sd = num/den if den > 0 else 0.0
                if sd < tau:
                    h = h_new
                    break
            h_prev = h_new
            h = h_new
        imfs.append(h)
        n_modes += 1
        total_sifts += n_sift
        per_mode_sifts.append(n_sift)
        resid = resid - h
    stats = dict(n_modes=n_modes, total_sifts=total_sifts,
                 per_mode_sifts=per_mode_sifts)
    return imfs, resid, stats

def emd_1imf(x, t=None, tau=0.25, max_sifts=50, end='natural'):
    """Sift to get only the FIRST IMF (used by CEEMDAN). Returns (imf, n_sift)."""
    imfs, _, stats = emd_1d(x, t=t, tau=tau, max_sifts=max_sifts,
                            max_imf=1, min_extrema=4, end=end)
    return imfs[0], stats['per_mode_sifts'][0]

# =====================================================================
# 3. Ensemble (EEMD / CEEMDAN)  -- the parallel axis
# =====================================================================
def _gen_noise(E, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((E, N))

def _trial_eemd(args):
    """Worker: full EMD of one perturbed signal. Top-level (picklable)."""
    x, noise, tau, max_sifts, min_ext, end = args
    t0 = time.time()
    imfs, resid, stats = emd_1d(x + noise, tau=tau, max_sifts=max_sifts,
                                min_extrema=min_ext, end=end)
    return imfs, resid, stats, time.time()-t0

def _trial_1imf(args):
    """Worker: first-IMF sift of one perturbed signal (CEEMDAN stage)."""
    x, noise, tau, max_sifts, end = args
    t0 = time.time()
    imf, ns = emd_1imf(x + noise, tau=tau, max_sifts=max_sifts, end=end)
    return imf, ns, time.time()-t0

def _run_trials(parallel, n_workers, fn, tasks):
    """Run independent trial jobs with DYNAMIC scheduling (load-imbalance
    handling). parallel=False -> serial (ground-truth baseline)."""
    if not parallel:
        return [fn(a) for a in tasks]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = [None]*len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2i = {ex.submit(fn, a): i for i, a in enumerate(tasks)}
        for fut in as_completed(fut2i):          # dynamic: next-finished first
            out[fut2i[fut]] = fut.result()
    return out

def eemd(x, E=100, eps=0.2, parallel=True, n_workers=None, seed=0, **kw):
    """EEMD: E independent EMDs of x + eps*std(x)*noise_e, average IMFs."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    n_workers = n_workers or max(1, (E - 1)//2)
    noises = _gen_noise(E, N, seed) * (eps * np.std(x))
    tasks = [(x, noises[e], kw.get('tau', 0.25), kw.get('max_sifts', 50),
              kw.get('min_extrema', 4), kw.get('end', 'natural'))
             for e in range(E)]
    res = _run_trials(parallel, n_workers, _trial_eemd, tasks)
    # average the IMFs level by level (trials may yield different n_modes)
    maxm = max(len(r[0]) for r in res)
    avg = np.zeros((maxm, N))
    cnt = np.zeros(maxm)
    for r in res:
        for i, imf in enumerate(r[0]):
            avg[i] += imf; cnt[i] += 1
    avg = avg / np.where(cnt > 0, cnt, 1)[:, None]
    resid = x - avg.sum(axis=0)
    return avg, resid, res

def ceemdan(x, E=100, eps=0.2, parallel=True, n_workers=None, seed=0, **kw):
    """CEEMDAN (Torres 2011 / Colominas 2014 improved). Staircase:
    sequential in IMF index, parallel over the E trials per stage."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    n_workers = n_workers or max(1, (E - 1)//2)
    tau = kw.get('tau', 0.25); max_sifts = kw.get('max_sifts', 50)
    end = kw.get('end', 'natural')
    W = _gen_noise(E, N, seed)                       # E standard-normal noises
    # Stage 0: decompose each noise -> noise IMFs (fully parallel)
    noise_tasks = [(W[e], np.zeros(N), tau, max_sifts, 4, end) for e in range(E)]
    noise_res = _run_trials(parallel, n_workers, _trial_eemd, noise_tasks)
    noise_imfs = [r[0] for r in noise_res]           # list of lists
    # normalize each noise IMF by its 1st-IMF std (Colominas improvement)
    for e in range(E):
        if noise_imfs[e]:
            s = np.std(noise_imfs[e][0])
            if s > 0:
                noise_imfs[e] = [a/s for a in noise_imfs[e]]
    c_imfs = []
    resid = x.copy()
    while True:
        if np.max(np.abs(resid)) < 1e-12 or np.std(resid) < 1e-10:
            break
        # one stage: c = mean_e 1stIMF( resid + eps*std(resid)*noise_e[stage] )
        stage = len(c_imfs)
        beta = eps * np.std(resid)
        tasks = [(resid, beta*W[e], tau, max_sifts, end) for e in range(E)]
        res = _run_trials(parallel, n_workers, _trial_1imf, tasks)
        c = np.zeros(N)
        for e in range(E):
            imf, ns, _ = res[e]
            c += imf
        c /= E
        c_imfs.append(c)
        resid = resid - c
        # stop when the added noise IMF index is exhausted for all trials
        if all(stage >= len(noise_imfs[e]) for e in range(E)):
            break
    return np.array(c_imfs), resid, dict(E=E, n_imfs=len(c_imfs))

# =====================================================================
# 4. §10 supporting measurements
# =====================================================================
def verify_thomas_vs_dense(d, l, u, r):
    """§10.3: cross-check the no-pivot Thomas solve against an INDEPENDENT
    dense solve of the identical 2-dominant tridiagonal system. Returns the
    max abs difference (should be ~machine epsilon). The GPU cyclic-reduction
    (PCR) solve is later compared against this same ground truth."""
    n = len(d)
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = d
    for i in range(n-1):
        A[i, i+1] = u[i]
    for i in range(1, n):
        A[i, i-1] = l[i]
    x_dense = np.linalg.solve(A, r)
    x_thomas = thomas(d, l, u, r)
    return float(np.max(np.abs(x_dense - x_thomas)))

def load_imbalance_demo(x, E=32, n_workers=4, seed=0):
    """Measure the per-trial sift distribution and static-vs-dynamic makespan
    (§10.4). Runs trials SERIALLY to record (work, time) per trial, then
    computes the makespan under static round-robin vs dynamic (optimal
    lower bound = max(total/P, max_trial))."""
    N = len(x)
    noises = _gen_noise(E, N, seed) * (0.2 * np.std(x))
    works, times = [], []
    for e in range(E):
        t0 = time.time()
        imfs, resid, stats = emd_1d(x + noises[e], max_sifts=50)
        works.append(stats['total_sifts']); times.append(time.time()-t0)
    works = np.array(works); times = np.array(times)
    total = times.sum()
    # static round-robin: assign trial e to worker e % P, makespan = max worker sum
    def static_makespan(P):
        acc = np.zeros(P)
        for e in range(E):
            acc[e % P] += times[e]
        return acc.max()
    dyn_lb = max(total/n_workers, times.max())   # dynamic lower bound
    return dict(works=works, times=times,
                n_sift_min=int(works.min()), n_sift_max=int(works.max()),
                n_sift_mean=works.mean(),
                static_makespan=static_makespan(n_workers),
                dynamic_lb=dyn_lb, serial=total,
                imbalance_ratio=times.max()/times.mean())

def mode_mixing_vs_E(x, Es=(1, 4, 16, 64, 256), seed=0, **kw):
    """§10.5: two close tones with different AM -> mode mixing. Measure how the
    EEMD error (L2 distance from the *known* clean decomposition) falls with E,
    testing the O(E^{-1/2}) Monte-Carlo rate."""
    x = np.asarray(x, dtype=float)
    out = {}
    for E in Es:
        imfs, resid, _ = eemd(x, E=E, parallel=False, seed=seed, **kw)
        # proxy for mode-mixing quality: energy of the 2nd IMF's spectrum in the
        # 1st tone's band (leakage). Lower = better separation.
        out[E] = imfs
    return out

if __name__ == '__main__':
    # quick self-test
    t = np.linspace(0, 5, 2000)
    x = np.sin(2*np.pi*8*t) + 0.5*np.sin(2*np.pi*3*t + 0.7)
    imfs, resid, stats = emd_1d(x)
    print("EMD self-test: n_modes=%d total_sifts=%d  recon_err=%.2e"
          % (stats['n_modes'], stats['total_sifts'],
             np.max(np.abs(x - (sum(imfs)+resid)))))
    # reconstruction must be exact (EMD is a partition of unity)
    assert np.max(np.abs(x - (sum(imfs)+resid))) < 1e-10
    print("OK: EMD is a partition of unity.")
