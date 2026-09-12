"""
phase0_microbench.py  --  PHASE-0 (throwaway, read-only vs shipped code)
=======================================================================
Hardware microbenchmarks for the block-Thomas + parallel-scan plan.
Run on BOTH machines (4070 now, GB10 Spark by the user) for a side-by-side.

Measures (the three numbers the plan's Amdahl model needs):
  1. FP64 DGEMM rate (cupy matmul)        -- essay §6 "first thing to measure"
  2. single-thread scalar FLOP rate       -- the "single-thread trap"
     (dependent add-multiply chain, clock64-timed)
  3. stream-scan rate: 1 thread vs 128 threads (one block = one SM),
     the exact production extrema-scan access pattern (3 loads/iter)
     -> directly quantifies the parallel-scan gain

clock64 is injected via a PTX @intrinsic (numba 0.67 exposes no device
clock), proven working in dbg_serial_split_v0.py.

NO shipped file is modified. Run:  .venv/Scripts/python.exe phase0_microbench.py
"""
import time
import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.extending import intrinsic
from llvmlite import ir

import warnings
from numba.core import errors as _numba_errors
warnings.filterwarnings('ignore', category=_numba_errors.NumbaPerformanceWarning)


@intrinsic
def clock64(typingctx):
    """Raw PTX: read the GPU's 64-bit cycle counter (per-SM, GHz-scaled)."""
    sig = types.int64()
    def codegen(context, builder, sig, args):
        fty = ir.FunctionType(ir.IntType(64), [])
        asm = ir.InlineAsm(fty, "mov.u64 $0, %clock64;", "=l", side_effect=True)
        return builder.call(asm, [])
    return sig, codegen


# ----------------------------------------------------------------------
# 2. single-thread dependent FLOP chain
# ----------------------------------------------------------------------
def _k_scalar(n):
    @cuda.jit
    def _kern(out, cyc):
        if cuda.threadIdx.x == 0:
            t0 = clock64()
            x = 1.0
            for i in range(n):
                x = x * 0.999999 + 1e-9
            out[0] = x
            cyc[0] = clock64() - t0
    return _kern


# ----------------------------------------------------------------------
# 3. stream scan, production access pattern: hv[i], hv[i-1], hv[i+1]
# ----------------------------------------------------------------------
def _k_scan1(N, nthreads):
    @cuda.jit
    def _kern(hv, out, cyc):
        tid = cuda.threadIdx.x
        if tid == 0:
            t0 = clock64()
            c = 0
            for i in range(1, N-1):
                if (hv[i] > hv[i-1]) and (hv[i] > hv[i+1]):
                    c += 1
            out[0] = c
            cyc[0] = clock64() - t0
    return _kern


def _k_scanN(N, nthreads):
    @cuda.jit
    def _kern(hv, out, cyc):
        tid = cuda.threadIdx.x
        if tid == 0:
            t0 = clock64()
        local = 0
        for i in range(tid+1, N-1, nthreads):
            if (hv[i] > hv[i-1]) and (hv[i] > hv[i+1]):
                local += 1
        # shared reduction
        red = cuda.shared.array(nthreads, dtype=np.int64)
        red[tid] = local
        cuda.syncthreads()
        step = nthreads // 2
        while step >= 1:
            if tid < step:
                red[tid] += red[tid+step]
            cuda.syncthreads()
            step //= 2
        if tid == 0:
            out[0] = red[0]
            cyc[0] = clock64() - t0
    return _kern


def bench(launch, reps=5):
    launch()
    cuda.synchronize()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        launch()
        cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print("device: %s  SMs=%d  L2=%.1f MB  clock=%.0f MHz"
          % (dev['name'].decode().strip('\x00'), dev['multiProcessorCount'],
             dev['l2CacheSize']/1024**2, dev['clockRate']/1000))

    N = 441000
    x = np.random.default_rng(0).standard_normal(N).astype(np.float64)
    hv = cuda.to_device(x)
    out = cuda.device_array(1, dtype=np.int64)
    cyc = cuda.device_array(1, dtype=np.int64)

    # ---- 1. FP64 DGEMM ----
    for m in (2048, 4096):
        a = cp.random.randn(m, m, dtype=np.float64)
        b = cp.random.randn(m, m, dtype=np.float64)
        cp.matmul(a, b)          # warm
        t0 = time.perf_counter()
        for _ in range(5):
            c = a @ b
        cp.cuda.stream.get_current_stream().synchronize()
        dt = (time.perf_counter() - t0) / 5
        tflops = 2 * m**3 / dt / 1e12
        print("\nFP64 matmul %dx%d: %.3f s  ->  %.3f TFLOPS" % (m, m, dt, tflops))

    # ---- 2. single-thread scalar FLOP rate ----
    n = 4_000_000
    k = _k_scalar(n)
    t = bench(lambda: k[1, 1](out, cyc))
    ccyc = int(np.asarray(cyc)[0])
    print("\nscalar dependent FLOP chain (1 thread, %d iters): %.3f s  "
          "-> %.2f Gflops  (%.2f cycles/iter)"
          % (n, t, 2*n/t/1e9, ccyc/n))

    # ---- 3. stream scan: 1 thread vs 128 threads (one SM) ----
    k1 = _k_scan1(N, 1)
    kN = _k_scanN(N, 128)
    t1 = bench(lambda: k1[1, 1](hv, out, cyc))
    c1 = int(np.asarray(cyc)[0])
    tN = bench(lambda: kN[1, 128](hv, out, cyc))
    cN = int(np.asarray(cyc)[0])
    print("\nstream scan (N=%d, production extrema pattern):" % N)
    print("  1 thread  : %.3f s  (%.3f cycles/elem)" % (t1, c1/(N-2)))
    print("  128 thr   : %.3f s  (%.3f cycles/elem)   ->  %.1fx scan speedup"
          % (tN, cN/(N-2), t1/tN))

    print("\n=== how this feeds the plan (T_block = 4.08 s @4070) ===")
    print("serial fraction = 95.8%%;  scans = 78.8%% of T_block,")
    print("thomas = 8.2%%, build = 8.3%%, mf = 0.6%%.")
    print("The scan speedup above is the lever for the 78.8%%.")


if __name__ == '__main__':
    main()
