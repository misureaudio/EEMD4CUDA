"""
test_block_thomas.py  --  PHASE-1 host-side reference + unit gates for the
Decay-Seamed Block Thomas (DSBT) + parallel-scan plan (BLOCK_THOMAS_PLAN.md).

Host-only: numpy only, NO GPU, NO kernel. This file is the "mirror of the
device algorithm exactly" the plan asks for, plus the three unit gates:

  U1  seam-window solver vs EXACT (emd_ref.thomas) and vs np.linalg.solve
      on real-sweep knots + uniform / CV0.4 / alternating-1:10 / CV1.5 grids.
      Gate: max|dM|/max|M| <= 1e-15 (and the raw boundary error is reported).
  U2  two-pass extrema scan (host emulation of the device algorithm) is
      bit-identical to the serial scan, incl. flat runs / plateaus / 1 extremum.
  U3  the Green's-function decay base r_i <= 2-sqrt(3) on all U1 knot sets
      (guards the whole seam-error argument against drift).

The seam-window solver below is a LITERAL translation of the device
`_window_thomas` helper (same B, S, same clipped windows, same dropped edge
couplings, same in-place dd/xx/cc bookkeeping). If this numpy mirror passes
U1, the device algorithm has the same math (modulo float64 associativity,
which is ~1 ulp and far below the gate).

Run:  .venv/Scripts/python.exe test_block_thomas.py
Exit: 0 = all gates pass; 1 = at least one failure.
"""
from __future__ import annotations
import math
import sys
import numpy as np

B = 250          # owned block size (plan §2.3)
S = 30           # halo width   (plan §2.3)
NTHR = 128       # threads per block (unchanged)
RHO_MAX = 2.0 - math.sqrt(3.0)      # 0.267949...  (2-dominant slowest decay)

FAILS = []

def check(name, cond, detail=""):
    print("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)


# =====================================================================
# Seam-window Thomas solver  (mirror of the device _window_thomas)
# =====================================================================
'''
def _window_thomas_np(d, l, u, r, x, c_, w0, w1):
    """Plain Thomas over the window [w0..w1] with the edge couplings DROPPED
    (row w0 uses d only, row w1 uses d only). In-place bookkeeping mirrors the
    device helper: the forward pass stores the modified diagonal in c_[w0..w1]
    (NOT overwriting d, because the next window's left halo re-reads d) and the
    modified rhs in x[w0..w1]; the back pass overwrites x with the solution.
    d, l, u, r are read-only."""
    if w1 < w0:
        return
    c_[w0] = 0.0
    x[w0] = r[w0] / d[w0]
    for i in range(w0 + 1, w1 + 1):
        m = l[i] / d[i]
        c_[i] = 0.0 if i == w1 else u[i] / (d[i] - l[i] * c_[i - 1])
        x[i] = (r[i] - l[i] * x[i - 1]) / (d[i] - l[i] * c_[i - 1])
    for i in range(w1 - 1, w0 - 1, -1):
        x[i] = x[i] - c_[i] * x[i + 1]
'''

def _window_thomas_np(d, l, u, r, x, c_, w0, w1):
    """Plain Thomas over the window [w0..w1] with the edge couplings DROPPED.
    Exactly mirrors the device _window_thomas."""
    if w1 < w0:
        return
    cprev = 0.0
    xprev = 0.0
    for i in range(w0, w1 + 1):
        denom = d[i] - l[i] * cprev
        cnew = 0.0 if i == w1 else u[i] / denom
        xnew = (r[i] - l[i] * xprev) / denom
        c_[i] = cnew
        x[i] = xnew
        cprev = cnew
        xprev = xnew
    for i in range(w1 - 1, w0 - 1, -1):
        x[i] = x[i] - c_[i] * x[i + 1]


def scan_extrema_2pass(arr, want_max, nthreads=NTHR):
    """Literal emulation of the device two-pass scan."""
    N = len(arr)
    idx = np.arange(1, N - 1)                      
    
    # pass 1: per-thread strided count
    cnt = np.zeros(nthreads, dtype=np.int64)
    for i in idx:
        t = (i - 1) % nthreads
        hit = (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]) if want_max \
              else (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1])
        if hit:
            cnt[t] += 1
            
    # Emulate the Hillis-Steele inclusive prefix sum safely
    v = np.cumsum(cnt)
    
    # pass 2: strided re-detect + compact
    total = int(v[-1]) if nthreads else 0
    out = np.zeros(total, dtype=np.int32)
    local = np.zeros(nthreads, dtype=np.int64)
    for i in idx:
        t = (i - 1) % nthreads
        hit = (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]) if want_max \
              else (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1])
        if hit:
            # Exclusive offset = inclusive_sum (v[t]) - local_total (cnt[t])
            off = v[t] - cnt[t]
            pos = int(off) + int(local[t])
            out[pos] = np.int32(i)
            local[t] += 1
            
    return out

def dsbt_solve(d, l, u, r, B=B, S=S):
    """Decay-Seamed Block Thomas: partition the nint interior unknowns into
    owned blocks of B; each block solves its clipped window [owned +/- S] as a
    standalone tridiagonal system (edge couplings dropped). Returns the full
    interior solution x (length nint)."""
    n = len(d)
    x = np.zeros(n)
    c_ = np.zeros(n)
    nb = (n + B - 1) // B
    for blk in range(nb):
        o0 = blk * B
        o1 = min(n, o0 + B) - 1
        w0 = max(0, o0 - S)
        w1 = min(n - 1, o1 + S)
        _window_thomas_np(d, l, u, r, x, c_, w0, w1)
    return x


def build_system(tm):
    """Build the 2-dominant interior system from knot times (ints) + values.
    Mirrors emd_ref._build_interior_system + _rhs exactly."""
    p = len(tm)
    H = np.diff(tm).astype(float)
    n = p - 2
    d = 2.0 * (H[0:n] + H[1:n + 1])
    l = np.zeros(n); u = np.zeros(n)
    l[1:] = H[1:n]
    u[0:n - 1] = H[1:n]
    hp = H[0:n]; hn = H[1:n + 1]
    return d, l, u, H, n


def rhs_of(tm, ym, H, n):
    hp = H[0:n]; hn = H[1:n + 1]
    yp = ym[0:n]; yk = ym[1:n + 1]; yn = ym[2:n + 2]
    return 6.0 * ((yn - yk) / hn - (yk - yp) / hp)


# =====================================================================
# U2: two-pass extrema scan (host emulation of the device _scan_extrema)
# =====================================================================
def scan_extrema_serial(arr, want_max):
    N = len(arr)
    kt = []
    for i in range(1, N - 1):
        if want_max:
            if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]):
                kt.append(i)
        else:
            if (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1]):
                kt.append(i)
    return np.array(kt, dtype=np.int32)

'''
def scan_extrema_2pass(arr, want_max, nthreads=NTHR):
    """Literal emulation of the device two-pass scan: pass 1 strided count +
    Hillis-Steele inclusive prefix sum, pass 2 strided re-detect + compact.
    Returns the compacted index array (time order, int32)."""
    N = len(arr)
    idx = np.arange(1, N - 1)                      # interior indices, time order
    # pass 1: per-thread strided count
    cnt = np.zeros(nthreads, dtype=np.int64)
    for i in idx:
        t = (i - 1) % nthreads
        hit = (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]) if want_max \
              else (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1])
        if hit:
            cnt[t] += 1
    # Hillis-Steele inclusive prefix sum (7 steps for 128)
    v = cnt.copy()
    step = 1
    while step < nthreads:
        for t in range(nthreads):
            if t + step < nthreads:
                v[t + step] = v[t + step] + v[t]
        step <<= 1
    # v[t] = inclusive count of hits with idx < idx[t] (i.e. exclusive offset)
    # pass 2: strided re-detect + compact
    total = int(v[-1]) if nthreads else 0
    out = np.zeros(total, dtype=np.int32)
    local = np.zeros(nthreads, dtype=np.int64)
    for i in idx:
        t = (i - 1) % nthreads
        hit = (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]) if want_max \
              else (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1])
        if hit:
            pos = int(v[t]) + int(local[t])
            out[pos] = np.int32(i)
            local[t] += 1
    return out
'''

def make_scan_cases():
    rng = np.random.default_rng(0)
    cases = {}
    cases['random'] = rng.standard_normal(5000)
    # flat runs (equal neighbours -> NOT an extremum, strict >)
    x = rng.standard_normal(4000)
    x[100:110] = x[100]
    x[200:260] = x[200]
    cases['flat_runs'] = x
    # plateau peak (a flat top: the plateau interior is NOT a strict extremum,
    # only the single sample that is > both neighbours counts)
    x = rng.standard_normal(4000)
    x[500:505] = 99.0
    cases['plateau'] = x
    # single extremum
    x = np.linspace(0, 1, 1000)
    x[500] = 5.0
    cases['single_max'] = x
    x = np.linspace(0, 1, 1000)
    x[500] = -5.0
    cases['single_min'] = x
    # all equal (zero extrema)
    cases['constant'] = np.ones(500)
    # monotone (zero extrema)
    cases['monotone'] = np.linspace(-3, 3, 2000)
    # alternating (every interior sample is an extremum)
    cases['alternating'] = (np.arange(1000) % 2) * 2.0 - 1.0
    return cases


# =====================================================================
# U3: Green's-function decay base
# =====================================================================
def decay_base(d, l, u, n):
    """Measure the geometric decay base of the interior Green's function.
    Solve A x = e_mid (mid = n//4, well inside the left half so the right
    boundary reflection is negligible), then fit log|x[i]| over i in
    [mid+5, mid+min(80, n-mid-5)] -> slope = -ln(1/r). Returns the base r."""
    if n < 100:
        return 0.0
    mid = n // 4
    rvec = np.zeros(n)
    rvec[mid] = 1.0
    x = np.linalg.solve(_dense(d, l, u, n), rvec)
    i0 = mid + 5
    i1 = mid + min(80, n - mid - 5)
    seg = x[i0:i1]
    if np.any(seg == 0) or np.any(seg < 0):
        seg = np.abs(seg)
    logseg = np.log(np.abs(seg))
    ii = np.arange(i0, i1)
    slope = np.polyfit(ii, logseg, 1)[0]
    return float(math.exp(-slope))


def _dense(d, l, u, n):
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = d
    for i in range(n - 1):
        A[i, i + 1] = u[i]
    for i in range(1, n):
        A[i, i - 1] = l[i]
    return A


# =====================================================================
# U1 knot sets
# =====================================================================
def real_sweep_knots(n_knots=5000, seed=0):
    """Real spline knots from the reference 10-octave sweep EMD (upper +
    lower), truncated to ~n_knots. Returns (tm, ym) as int sample indices +
    float values, exactly as the GPU stores them."""
    from refsignal import make_ref_signal
    from emd_ref import envelope
    x, t = make_ref_signal()
    N = min(44100, len(x))
    x = x[:N]
    tt = np.arange(N)
    tm, ym = [], []
    # first few IMFs give the densest, most representative knot sets
    resid = x.copy()
    for m in range(4):
        eu, _ = envelope(tt, resid, 'max', 'natural')
        el, _ = envelope(tt, resid, 'min', 'natural')
        # re-derive the knot indices the same way the kernel does
        kmax = np.where((resid[1:-1] > resid[:-2]) & (resid[1:-1] > resid[2:]))[0] + 1
        kmin = np.where((resid[1:-1] < resid[:-2]) & (resid[1:-1] < resid[2:]))[0] + 1
        for k in (kmax, kmin):
            if len(k) >= 4:
                tm.append(k); ym.append(resid[k])
        h = resid - 0.5 * (eu + el)
        resid = resid - h
        if sum(len(a) for a in tm) > n_knots:
            break
    # take the first knot set long enough (>= 4 knots)
    for k, y in zip(tm, ym):
        if len(k) >= 64:
            return k.astype(np.int64), y.astype(float)
    raise RuntimeError("no real knot set long enough")


def synth_knots(kind, p=5000, seed=1):
    rng = np.random.default_rng(seed)
    if kind == 'uniform':
        tm = np.arange(p) * 10
    elif kind == 'cv04':
        h = 10.0 * (1.0 + 0.4 * rng.standard_normal(p - 1))
        h = np.clip(h, 0.2, None)
        tm = np.concatenate(([0], np.cumsum(h)))
    elif kind == 'alt110':
        h = np.repeat([1.0, 10.0], (p - 1) // 2 + 1)[:p - 1]
        tm = np.concatenate(([0], np.cumsum(h)))
    elif kind == 'cv15':
        h = 10.0 * np.exp(1.5 * rng.standard_normal(p - 1) * 0.5)
        h = np.clip(h, 0.2, None)
        tm = np.concatenate(([0], np.cumsum(h)))
    else:
        raise ValueError(kind)
    tm = tm[:p].astype(np.int64)
    # make strictly increasing (clip can produce ties)
    for i in range(1, len(tm)):
        if tm[i] <= tm[i - 1]:
            tm[i] = tm[i - 1] + 1
    ym = rng.standard_normal(len(tm))
    return tm, ym


# =====================================================================
def main():
    print("=== Phase 1: DSBT host-side reference + unit gates ===")
    print("B=%d  S=%d  RHO_MAX(2-sqrt3)=%.6f" % (B, S, RHO_MAX))

    # ---------------- U1: solver vs exact ----------------
    print("\n--- U1: seam-window solver vs exact Thomas / dense ---")
    u1_max = 0.0
    u1_bound_max = 0.0
    sets = [('real-sweep',) + real_sweep_knots(),
            ('uniform',) + synth_knots('uniform'),
            ('cv04',) + synth_knots('cv04'),
            ('alt110',) + synth_knots('alt110'),
            ('cv15',) + synth_knots('cv15')]
    bases = {}
    for name, tm, ym in sets:
        d, l, u, H, n = build_system(tm)
        r = rhs_of(tm, ym, H, n)
        x_ex = np.linalg.solve(_dense(d, l, u, n), r)   # independent exact
        import emd_ref as R
        x_th = R.thomas(d, l, u, r)                      # exact Thomas
        x_ds = dsbt_solve(d, l, u, r)
        scale = max(float(np.max(np.abs(x_ex))), 1e-300)
        rel = float(np.max(np.abs(x_ds - x_ex))) / scale
        # raw boundary error (first & last interior unknowns = the seam rims)
        bound = max(abs(x_ds[0] - x_ex[0]), abs(x_ds[-1] - x_ex[-1])) / scale
        u1_max = max(u1_max, rel)
        u1_bound_max = max(u1_bound_max, bound)
        bases[name] = decay_base(d, l, u, n)
        ok = rel <= 1e-15
        print("  %-10s n=%5d  rel=%.2e  boundary=%.2e  %s"
              % (name, n, rel, bound, "PASS" if ok else "FAIL"))
        if not ok:
            FAILS.append("U1:%s" % name)
    check("U1 solver vs exact (all sets) <= 1e-15",
          u1_max <= 1e-15, "max rel=%.2e  (boundary max=%.2e)"
          % (u1_max, u1_bound_max))

    # ---------------- U2: scan equivalence ----------------
    print("\n--- U2: two-pass scan bit-identical to serial ---")
    u2_ok = True
    for name, arr in make_scan_cases().items():
        for want_max in (True, False):
            tag = "max" if want_max else "min"
            ref = scan_extrema_serial(arr, want_max)
            got = scan_extrema_2pass(arr, want_max)
            same = (ref.shape == got.shape) and (
                (ref == got).all() if ref.size else True)
            if not same:
                u2_ok = False
                print("  [FAIL] %s/%s  ref=%d got=%d"
                      % (name, tag, len(ref), len(got)))
    check("U2 scan equivalence (all cases, max+min)", u2_ok,
          "%d cases x 2" % len(make_scan_cases()))

    # ---------------- U3: decay base ----------------
    print("\n--- U3: Green's-function decay base <= 2-sqrt(3) ---")
    u3_ok = True
    for name, r in bases.items():
        ok = r <= RHO_MAX + 1e-9
        if not ok:
            u3_ok = False
        print("  %-10s  r=%.6f  %s" % (name, r, "PASS" if ok else "FAIL"))
    check("U3 decay base <= 2-sqrt(3) (all sets)", u3_ok,
          "max r=%.6f" % max(bases.values()))

    # ---------------- summary ----------------
    print("\n=== SUMMARY ===")
    if FAILS:
        print("  FAILED: %s" % ", ".join(FAILS))
        return 1
    print("  All Phase-1 unit gates PASSED (U1, U2, U3).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
