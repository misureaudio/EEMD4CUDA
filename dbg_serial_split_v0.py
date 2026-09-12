"""
dbg_serial_split.py  --  PHASE-0 (throwaway, read-only vs shipped code)
=====================================================================
Measure the per-phase time split of the current production kernel at
N=441000, E=1 (the T_block benchmark), so we know what fraction of the
thread-0 serial work is (a) extrema scans, (b) H/system fills, (c) the
Thomas solve, (d) Mf fills, vs (e) the already-parallel eval/SD/store.

Method: a faithful copy of emd_gpu._kern with clock64() timestamps
around each phase; per-phase cycle deltas accumulate into a global int64
counter. One real run (identical dynamics, identical stop logic) -> the
breakdown of the actual T_block, not a model.

NO shipped file is imported-modified or written. This file is self-contained.

Run:  .venv/Scripts/python.exe dbg_serial_split.py
"""
import time
import numpy as np
from numba import cuda

from refsignal import make_ref_signal

from numba.core import types
from numba.cuda.extending import intrinsic
from llvmlite import ir

# EMD kernels launch one block per trial (grid = E); small E is intentional,
# so silence numba's "low occupancy" performance warning.
import warnings
from numba.core import errors as _numba_errors
warnings.filterwarnings('ignore', category=_numba_errors.NumbaPerformanceWarning)

@intrinsic
def clock64(typingctx):
    """Injects raw PTX assembly to read the GPU's 64-bit cycle counter."""
    sig = types.int64()
    def codegen(context, builder, sig, args):
        # Define LLVM function type: returns 64-bit int, takes no args
        fty = ir.FunctionType(ir.IntType(64), [])
        # Inline PTX: move special register %clock64 into output constraint '=l' (64-bit int)
        asm = ir.InlineAsm(fty, "mov.u64 $0, %clock64;", "=l", side_effect=True)
        return builder.call(asm, [])
    return sig, codegen

@cuda.jit(device=True, inline=True)
def _thomas(d, l, u, r, x, c_, n):
    if n <= 0:
        return
    c_[0] = (u[0] / d[0]) if n > 1 else 0.0
    x[0] = r[0] / d[0]
    for i in range(1, n):
        c_[i] = (u[i] / (d[i] - l[i]*c_[i-1])) if i < n-1 else 0.0
        x[i] = (r[i] - l[i]*x[i-1]) / (d[i] - l[i]*c_[i-1])
    for i in range(n-2, -1, -1):
        x[i] = x[i] - c_[i]*x[i+1]


@cuda.jit(device=True, inline=True)
def _spline_eval_bin(tm, ym, M, t, p):
    if p < 2:
        return ym[0] if p == 1 else 0.0
    lo = 0
    hi = p - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if tm[mid] <= t:
            lo = mid
        else:
            hi = mid
    i = lo
    if i < 0:
        i = 0
    if i > p - 2:
        i = p - 2
    h = float(tm[i+1] - tm[i])
    if h <= 0.0:
        return ym[i]
    a = (float(tm[i+1]) - t) / h
    b = (t - float(tm[i])) / h
    return (a*ym[i] + b*ym[i+1]
            + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h) / 6.0)


# phase buckets
P_RESID_COUNT = 0
P_MAX_SCAN    = 1
P_UP_BUILD    = 2
P_UP_THOMAS   = 3
P_UP_MF       = 4
P_MIN_SCAN    = 5
P_LO_BUILD    = 6
P_LO_THOMAS   = 7
P_LO_MF       = 8
P_PAR_EVAL    = 9
P_PAR_STORE   = 10
P_NPHASE      = 11


def _make_instrumented(N, K, MAXIMF, nthreads):
    @cuda.jit
    def _kern(S, resid, h, out, eu, W, W_int, nsift, nmode, ph,
              tau, max_sifts, min_extrema):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        sig = S[b, :]
        rs  = resid[b, :]
        hv  = h[b, :]
        eur = eu[b, :]
        kt = W_int[b, :]
        ky = W[b, 0*K:1*K]
        H  = W[b, 1*K:2*K]
        dd = W[b, 2*K:3*K]
        ll = W[b, 3*K:4*K]
        uu = W[b, 4*K:5*K]
        rr = W[b, 5*K:6*K]
        xx = W[b, 6*K:7*K]
        cc = W[b, 7*K:8*K]
        Mf = W[b, 8*K:9*K]
        red = cuda.shared.array(nthreads, dtype=np.float64)
        nm_s = cuda.shared.array(1, dtype=np.int32)
        nn_s = cuda.shared.array(1, dtype=np.int32)
        stop = cuda.shared.array(1, dtype=np.int32)

        # only thread 0 accumulates phase cycles for this block
        if tid == 0:
            t0 = clock64()

        for i in range(tid, N, nthreads):
            rs[i] = sig[i]
            hv[i] = sig[i]

        total_sifts = 0
        nmode_b = 0
        for m in range(MAXIMF):
            cuda.syncthreads()
            # ---- residual extrema count (thread 0) ----
            if tid == 0:
                tc = clock64()
                nm = 0; nn = 0
                for i in range(1, N-1):
                    if rs[i] > rs[i-1]:
                        if rs[i] > rs[i+1]:
                            nm += 1
                    if rs[i] < rs[i-1]:
                        if rs[i] < rs[i+1]:
                            nn += 1
                ph[P_RESID_COUNT] += (clock64() - tc)
                nm_s[0] = nm; nn_s[0] = nn
            cuda.syncthreads()
            nm = nm_s[0]; nn = nn_s[0]
            if (nm + nn) < min_extrema or nm < 2 or nn < 2:
                break

            for i in range(tid, N, nthreads):
                hv[i] = rs[i]
            stop[0] = 0
            nsift_local = 0
            for s in range(max_sifts):
                cuda.syncthreads()
                if stop[0] != 0:
                    break
                # ---- MAX scan (thread 0) ----
                if tid == 0:
                    tc = clock64()
                    nmax = 0
                    for i in range(1, N-1):
                        if (hv[i] > hv[i-1]) and (hv[i] > hv[i+1]):
                            if nmax < K:
                                kt[nmax] = np.int32(i)
                                ky[nmax] = hv[i]
                            nmax += 1
                    ph[P_MAX_SCAN] += (clock64() - tc)
                    nm_s[0] = nmax
                cuda.syncthreads()
                nm = nm_s[0]
                if nm < 2:
                    break
                # ---- UPPER build (thread 0) ----
                if tid == 0:
                    tc = clock64()
                    for j in range(nm-1):
                        H[j] = float(kt[j+1] - kt[j])
                    nint = nm - 2
                    for q in range(nint):
                        dd[q] = 2.0*(H[q] + H[q+1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint-1 else H[q+1]
                        rr[q] = 6.0*((ky[q+2]-ky[q+1])/H[q+1] - (ky[q+1]-ky[q])/H[q])
                    ph[P_UP_BUILD] += (clock64() - tc)
                    tc = clock64()
                    _thomas(dd, ll, uu, rr, xx, cc, nint)
                    ph[P_UP_THOMAS] += (clock64() - tc)
                    tc = clock64()
                    Mf[0] = 0.0; Mf[nm-1] = 0.0
                    for i in range(nint):
                        Mf[i+1] = xx[i]
                    ph[P_UP_MF] += (clock64() - tc)
                cuda.syncthreads()
                # ---- parallel: upper eval ----
                for i in range(tid, N, nthreads):
                    eur[i] = _spline_eval_bin(kt, ky, Mf, i, nm)
                cuda.syncthreads()
                # ---- MIN scan (thread 0) ----
                if tid == 0:
                    tc = clock64()
                    nmin = 0
                    for i in range(1, N-1):
                        if (hv[i] < hv[i-1]) and (hv[i] < hv[i+1]):
                            if nmin < K:
                                kt[nmin] = np.int32(i)
                                ky[nmin] = hv[i]
                            nmin += 1
                    ph[P_MIN_SCAN] += (clock64() - tc)
                    nn_s[0] = nmin
                cuda.syncthreads()
                nn = nn_s[0]
                if nn < 2:
                    break
                # ---- LOWER build + thomas + mf (thread 0) ----
                if tid == 0:
                    tc = clock64()
                    for j in range(nn-1):
                        H[j] = float(kt[j+1] - kt[j])
                    nint = nn - 2
                    for q in range(nint):
                        dd[q] = 2.0*(H[q] + H[q+1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint-1 else H[q+1]
                        rr[q] = 6.0*((ky[q+2]-ky[q+1])/H[q+1] - (ky[q+1]-ky[q])/H[q])
                    ph[P_LO_BUILD] += (clock64() - tc)
                    tc = clock64()
                    _thomas(dd, ll, uu, rr, xx, cc, nint)
                    ph[P_LO_THOMAS] += (clock64() - tc)
                    tc = clock64()
                    Mf[0] = 0.0; Mf[nn-1] = 0.0
                    for i in range(nint):
                        Mf[i+1] = xx[i]
                    ph[P_LO_MF] += (clock64() - tc)
                cuda.syncthreads()
                # ---- parallel: lower eval + mean + subtract + SD ----
                tc_par = clock64() if tid == 0 else 0
                num = 0.0; den = 0.0
                for i in range(tid, N, nthreads):
                    el_i = _spline_eval_bin(kt, ky, Mf, i, nn)
                    mval = 0.5*(eur[i] + el_i)
                    hnew = hv[i] - mval
                    num += mval*mval
                    den += hv[i]*hv[i]
                    hv[i] = hnew
                red[tid] = num
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step:
                        red[tid] += red[tid+step]
                    cuda.syncthreads()
                    step //= 2
                numsum = red[0]
                cuda.syncthreads()
                red[tid] = den
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step:
                        red[tid] += red[tid+step]
                    cuda.syncthreads()
                    step //= 2
                densum = red[0]
                if tid == 0:
                    ph[P_PAR_EVAL] += (clock64() - tc_par)
                nsift_local += 1
                if s >= 1:
                    sd = (numsum/densum) if densum > 0 else 0.0
                    if sd < tau:
                        if tid == 0:
                            stop[0] = 1
                        break
            total_sifts += nsift_local
            # ---- parallel: store IMF + residual update ----
            tc_par = clock64() if tid == 0 else 0
            outrow = out[b, m, :]
            for i in range(tid, N, nthreads):
                outrow[i] = hv[i]
                rs[i] = rs[i] - hv[i]
            if tid == 0:
                ph[P_PAR_STORE] += (clock64() - tc_par)
            nmode_b += 1

        if tid == 0:
            nsift[b] = total_sifts
            nmode[b] = nmode_b

    return _kern


def main():
    xref, _ = make_ref_signal()
    N = len(xref)
    K = max(16, N // 2)
    MAXIMF = 16
    nthreads = 128
    kern = _make_instrumented(N, K, MAXIMF, nthreads)

    x = xref[None, :].astype(np.float64)
    S = cuda.to_device(np.ascontiguousarray(x))
    resid = cuda.device_array((1, N), dtype=np.float64)
    h     = cuda.device_array((1, N), dtype=np.float64)
    out   = cuda.device_array((1, MAXIMF, N), dtype=np.float64)
    eu    = cuda.device_array((1, N), dtype=np.float64)
    W     = cuda.device_array((1, 9*K), dtype=np.float64)
    W_int = cuda.device_array((1, K), dtype=np.int32)
    nsift = cuda.device_array(1, dtype=np.int32)
    nmode = cuda.device_array(1, dtype=np.int32)
    ph    = cuda.device_array(P_NPHASE, dtype=np.int64)

    # warm JIT
    ph = cuda.to_device(np.zeros(P_NPHASE, dtype=np.int64))
    kern[1, nthreads](S, resid, h, out, eu, W, W_int, nsift, nmode, ph,
                      0.25, 50, 4)
    cuda.synchronize()

    # timed run
    ph = cuda.to_device(np.zeros(P_NPHASE, dtype=np.int64))
    t0 = time.perf_counter()
    kern[1, nthreads](S, resid, h, out, eu, W, W_int, nsift, nmode, ph,
                      0.25, 50, 4)
    cuda.synchronize()
    wall = time.perf_counter() - t0

    ph_np = np.asarray(ph)
    ns = int(np.asarray(nsift)[0]); nm = int(np.asarray(nmode)[0])
    print("N=%d  T_block wall=%.3fs  nmode=%d  total_sifts=%d"
          % (N, wall, nm, ns))
    print("\nper-phase cycle breakdown (thread-0 clock64):")
    names = ["resid_count (serial, per-IMF)",
             "MAX scan (serial)",
             "UPPER build (serial)",
             "UPPER thomas (serial)",
             "UPPER Mf fill (serial)",
             "MIN scan (serial)",
             "LOWER build (serial)",
             "LOWER thomas (serial)",
             "LOWER Mf fill (serial)",
             "PAR eval+SD (128 thr)",
             "PAR store (128 thr)"]
    tot = ph_np.sum()
    rows = []
    for i in range(P_NPHASE):
        rows.append((ph_np[i], names[i], i))
    rows.sort(reverse=True)
    serial = 0
    for cyc, name, i in rows:
        frac = 100.0 * cyc / tot if tot else 0
        tag = "serial" if i < P_PAR_EVAL else "parallel"
        if i < P_PAR_EVAL:
            serial += cyc
        print("  %22s : %12.3f ms  %5.1f%%   [%s]"
              % (name, cyc/1e6, frac, tag))
    print("\n  SERIAL total   : %12.3f ms  %5.1f%% of phase-time"
          % (serial/1e6, 100.0*serial/tot))
    print("  PARALLEL total : %12.3f ms  %5.1f%% of phase-time"
          % ((tot-serial)/1e6, 100.0*(tot-serial)/tot))
    print("  (phase-time = sum of measured phases; wall includes launch/gaps)")


if __name__ == '__main__':
    main()
