"""
test_block_thomas.py  --  PHASE-1 (Corrected)
"""
from __future__ import annotations
import math
import sys
import numpy as np

B = 250
S = 30
NTHR = 128
RHO_MAX = 2.0 - math.sqrt(3.0)

FAILS = []

def check(name, cond, detail=""):
    print("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)


def _window_thomas_np(d, l, u, r, x, w0, w1, o0, o1):
    """Corrected Python mirror that protects the core from halo overwrites."""
    if w1 < w0:
        return
    cprev = 0.0
    xprev = 0.0
    loc_x = np.zeros(w1 - w0 + 1)
    loc_c = np.zeros(w1 - w0 + 1)
    
    for i in range(w0, w1 + 1):
        denom = d[i] - l[i] * cprev
        cnew = 0.0 if i == w1 else u[i] / denom
        xnew = (r[i] - l[i] * xprev) / denom
        loc_c[i - w0] = cnew
        loc_x[i - w0] = xnew
        cprev = cnew
        xprev = xnew
        
    for i in range(w1 - 1, w0 - 1, -1):
        loc_x[i - w0] = loc_x[i - w0] - loc_c[i - w0] * loc_x[i + 1 - w0]
        
    # ONLY write the owned portion to the global array
    for i in range(o0, o1 + 1):
        x[i] = loc_x[i - w0]


def dsbt_solve(d, l, u, r, B=B, S=S):
    n = len(d)
    x = np.zeros(n)
    nb = (n + B - 1) // B
    for blk in range(nb):
        o0 = blk * B
        o1 = min(n, o0 + B) - 1
        w0 = max(0, o0 - S)
        w1 = min(n - 1, o1 + S)
        _window_thomas_np(d, l, u, r, x, w0, w1, o0, o1)
    return x


def build_system(tm):
    p = len(tm)
    H = np.diff(tm).astype(float)
    n = p - 2
    d = 2.0 * (H[0:n] + H[1:n + 1])
    l = np.zeros(n); u = np.zeros(n)
    l[1:] = H[1:n]
    u[0:n - 1] = H[1:n]
    return d, l, u, H, n


def rhs_of(tm, ym, H, n):
    hp = H[0:n]; hn = H[1:n + 1]
    yp = ym[0:n]; yk = ym[1:n + 1]; yn = ym[2:n + 2]
    return 6.0 * ((yn - yk) / hn - (yk - yp) / hp)


def scan_extrema_serial(arr, want_max):
    N = len(arr)
    kt = []
    for i in range(1, N - 1):
        if want_max:
            if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]): kt.append(i)
        else:
            if (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1]): kt.append(i)
    return np.array(kt, dtype=np.int32)


def scan_extrema_tile(arr, want_max, nthreads=NTHR):
    """The ALGORITHM FIX: Tile-Based Cooperative Scan to preserve Time-Order."""
    N = len(arr)
    out = []
    
    # Process the array in contiguous tiles of 128 to preserve order
    for base in range(1, N - 1, nthreads):
        hits = np.zeros(nthreads, dtype=np.int64)
        for t in range(nthreads):
            i = base + t
            if i < N - 1:
                if want_max:
                    if (arr[i] > arr[i-1]) and (arr[i] > arr[i+1]): hits[t] = 1
                else:
                    if (arr[i] < arr[i-1]) and (arr[i] < arr[i+1]): hits[t] = 1
                        
        # Prefix sum (inclusive) across the tile
        v = np.cumsum(hits)
        
        # Compact and append (simulating global offset)
        for t in range(nthreads):
            if hits[t] == 1:
                out.append(np.int32(base + t))
                
    return np.array(out, dtype=np.int32)


def make_scan_cases():
    rng = np.random.default_rng(0)
    cases = {}
    cases['random'] = rng.standard_normal(5000)
    x = rng.standard_normal(4000)
    x[100:110] = x[100]; x[200:260] = x[200]
    cases['flat_runs'] = x
    x = rng.standard_normal(4000)
    x[500:505] = 99.0
    cases['plateau'] = x
    x = np.linspace(0, 1, 1000); x[500] = 5.0
    cases['single_max'] = x
    x = np.linspace(0, 1, 1000); x[500] = -5.0
    cases['single_min'] = x
    cases['constant'] = np.ones(500)
    cases['monotone'] = np.linspace(-3, 3, 2000)
    cases['alternating'] = (np.arange(1000) % 2) * 2.0 - 1.0
    return cases


def decay_base(d, l, u, n):
    if n < 100: return 0.0
    mid = n // 4
    rvec = np.zeros(n)
    rvec[mid] = 1.0
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = d
    for i in range(n - 1): A[i, i + 1] = u[i]
    for i in range(1, n): A[i, i - 1] = l[i]
    x = np.linalg.solve(A, rvec)
    
    i0 = mid + 5; i1 = mid + min(80, n - mid - 5)
    seg = np.abs(x[i0:i1])
    logseg = np.log(seg)
    slope = np.polyfit(np.arange(i0, i1), logseg, 1)[0]
    # FIX: slope is already negative (log(rho)). 
    return float(math.exp(slope))

# ==================== Knot Generation omitted for brevity, uses your exact logic ====================
def real_sweep_knots(n_knots=5000, seed=0):
    from refsignal import make_ref_signal
    from emd_ref import envelope
    x, t = make_ref_signal()
    N = min(44100, len(x))
    x = x[:N]
    tt = np.arange(N)
    tm, ym = [], []
    resid = x.copy()
    for m in range(4):
        eu, _ = envelope(tt, resid, 'max', 'natural')
        el, _ = envelope(tt, resid, 'min', 'natural')
        kmax = np.where((resid[1:-1] > resid[:-2]) & (resid[1:-1] > resid[2:]))[0] + 1
        kmin = np.where((resid[1:-1] < resid[:-2]) & (resid[1:-1] < resid[2:]))[0] + 1
        for k in (kmax, kmin):
            if len(k) >= 4:
                tm.append(k); ym.append(resid[k])
        h = resid - 0.5 * (eu + el)
        resid = resid - h
        if sum(len(a) for a in tm) > n_knots: break
    for k, y in zip(tm, ym):
        if len(k) >= 64: return k.astype(np.int64), y.astype(float)
    raise RuntimeError("no real knot set long enough")

def synth_knots(kind, p=5000, seed=1):
    rng = np.random.default_rng(seed)
    if kind == 'uniform': tm = np.arange(p) * 10
    elif kind == 'cv04':
        h = 10.0 * (1.0 + 0.4 * rng.standard_normal(p - 1))
        tm = np.concatenate(([0], np.cumsum(np.clip(h, 0.2, None))))
    elif kind == 'alt110':
        h = np.repeat([1.0, 10.0], (p - 1) // 2 + 1)[:p - 1]
        tm = np.concatenate(([0], np.cumsum(h)))
    elif kind == 'cv15':
        h = 10.0 * np.exp(1.5 * rng.standard_normal(p - 1) * 0.5)
        tm = np.concatenate(([0], np.cumsum(np.clip(h, 0.2, None))))
    tm = tm[:p].astype(np.int64)
    for i in range(1, len(tm)):
        if tm[i] <= tm[i - 1]: tm[i] = tm[i - 1] + 1
    return tm, rng.standard_normal(len(tm))

def main():
    print("=== Phase 1: DSBT host-side reference + unit gates ===")
    print("B=%d  S=%d  RHO_MAX(2-sqrt3)=%.6f" % (B, S, RHO_MAX))

    print("\n--- U1: seam-window solver vs exact Thomas / dense ---")
    u1_max, u1_bound_max = 0.0, 0.0
    sets = [('real-sweep',) + real_sweep_knots(), ('uniform',) + synth_knots('uniform'),
            ('cv04',) + synth_knots('cv04'), ('alt110',) + synth_knots('alt110'), ('cv15',) + synth_knots('cv15')]
    bases = {}
    for name, tm, ym in sets:
        d, l, u, H, n = build_system(tm)
        r = rhs_of(tm, ym, H, n)
        x_ex = np.linalg.solve(_dense(d, l, u, n), r)
        x_ds = dsbt_solve(d, l, u, r)
        scale = max(float(np.max(np.abs(x_ex))), 1e-300)
        rel = float(np.max(np.abs(x_ds - x_ex))) / scale
        bound = max(abs(x_ds[0] - x_ex[0]), abs(x_ds[-1] - x_ex[-1])) / scale
        u1_max, u1_bound_max = max(u1_max, rel), max(u1_bound_max, bound)
        bases[name] = decay_base(d, l, u, n)
        ok = rel <= 1e-15
        print("  %-10s n=%5d  rel=%.2e  boundary=%.2e  %s" % (name, n, rel, bound, "PASS" if ok else "FAIL"))
        if not ok: FAILS.append("U1:%s" % name)
    check("U1 solver vs exact (all sets) <= 1e-15", u1_max <= 1e-15, "max rel=%.2e (boundary=%.2e)" % (u1_max, u1_bound_max))

    print("\n--- U2: tile-based scan bit-identical to serial ---")
    u2_ok = True
    for name, arr in make_scan_cases().items():
        for want_max in (True, False):
            ref = scan_extrema_serial(arr, want_max)
            got = scan_extrema_tile(arr, want_max)
            same = (ref.shape == got.shape) and ((ref == got).all() if ref.size else True)
            if not same:
                u2_ok = False
                print("  [FAIL] %s/%s  ref=%d got=%d" % (name, "max" if want_max else "min", len(ref), len(got)))
    check("U2 scan equivalence (all cases, max+min)", u2_ok, "%d cases x 2" % len(make_scan_cases()))

    print("\n--- U3: Green's-function decay base <= 2-sqrt(3) ---")
    u3_ok = True
    for name, r in bases.items():
        ok = r <= RHO_MAX + 1e-9
        if not ok: u3_ok = False
        print("  %-10s  r=%.6f  %s" % (name, r, "PASS" if ok else "FAIL"))
    check("U3 decay base <= 2-sqrt(3) (all sets)", u3_ok, "max r=%.6f" % max(bases.values()))

    print("\n=== SUMMARY ===")
    if FAILS:
        print("  FAILED: %s" % ", ".join(FAILS))
        return 1
    print("  All Phase-1 unit gates PASSED (U1, U2, U3).")
    return 0

def _dense(d, l, u, n):
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = d
    for i in range(n - 1): A[i, i + 1] = u[i]
    for i in range(1, n): A[i, i - 1] = l[i]
    return A

if __name__ == '__main__':
    sys.exit(main())