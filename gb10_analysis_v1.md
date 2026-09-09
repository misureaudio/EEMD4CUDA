# Forecast: Running `emd_par.py` on an NVIDIA GB10 (DGX Spark)

*Companion to the EMD/EEMD parallel essay. Audience: signal / numerical / mathematical-physics
experts. Conventions: **verified** = measured on the RTX A2000 (GA106, CC 8.6); **spec** = from
NVIDIA datasheets / Hot Chips / ISSCC; **forecast** = derived from the verified per-block model.
One number is a genuine unknown and is flagged as such.*

---

## 0. A correction to the essay (and to the review)

The essay's §6(b) read the flat 7.0–7.1 s wall time (E = 8 → 128) as the GPU "saturating at
≈8–10 concurrent blocks." That reading is **wrong**, and the corrected model changes the
GB10 forecast materially.

> **Source of the error.** The "flat 7.0 s" came from a benchmark that passed a *one‑
> dimensional* signal with `E=64/128`; a bug in `gpu_eemd_full` silently reset `E=1` for a
> 1‑D input, so every "E=64" row was in fact **one block**. The API now raises on 1‑D + E>1,
> and the corrected measurement uses a true `(E, N)` array. The "7.0 s" was also from a
> lower‑extrema synthetic chirp, not the reference sweep.

Measured on the A2000, reference sweep (warm, true `(E,N)` input):

| N | E=1 warm (per‑block time T_blk) | wall vs E |
|---|---|---|
| 441 000 | **9.23 s** | 10.0 s (E=8, ≈1.1 waves), 20.1 s (E=32, ≈2.2), 21.8 s (E=64, ≈2.4) |
| 110 250 | **2.18 s** | 2.71 s (E=8), 2.91 s (E=32, ≈1.3), 8.39 s (E=128, ≈3.9), 14.66 s (E=256, ≈6.7) |

So the A2000 runs **tens of blocks in parallel** (not "saturates at 8–10"), and the wall
time grows in **waves** as E exceeds the resident capacity. The resident capacity C is set
primarily by **device memory** (each block holds its signal + working arrays + output IMFs,
≈21 MB at N=441k) and secondarily by per‑SM occupancy — it is *not* a hard SM‑count cap.

What sets the per‑block time is the **thread‑0 serial fraction**: the extremum scan
(O(N), one thread) and the Thomas solve (O(n), n ≈ N/2, one thread). It scales ~linearly in N
(2.18 s at 110k → 9.23 s at 441k, ratio 4.2 ≈ 441/110 = 4.0, the small excess being fixed
overhead). The other 127 threads of the block run the O(N) parallel phases (spline eval,
subtract, SD reduction), but those are **hidden** behind the serial fraction. This is an
Amdahl bottleneck, and it is **hardware‑agnostic**: it is one thread doing O(N) work,
independent of how many cores the GPU has.

Also corrected: the A2000 is **GA106** (3328 CUDA cores, 26 SMs, 6 MB L2, 288 GB/s, 12 GB
GDDR6), not GA102. The 1:64 FP64 restriction is confirmed (0.1248 TFLOPS FP64 vs 7.987
TFLOPS FP32). The essay's §6(d) "held hostage by market segmentation" is correct in kind;
the chip name was wrong.

This correction matters for the GB10 because it relocates the bottleneck: the GB10's extra
cores and shared memory do **not** touch the thread‑0 serial fraction; they touch
*concurrency* and *transfer*, which is exactly where the GB10's 128 GB and NVLink‑C2C pay.

---

## 1. GB10 verified specifications

| property | GB10 (DGX Spark) | A2000 (measured device) |
|---|---|---|
| architecture | Blackwell iGPU (G‑dielet, TSMC 3 nm) | Ampere GA106 |
| compute capability | **12.0 (sm_120)** | 8.6 |
| CUDA cores | **6 144** | 3 328 |
| SMs | ~48 (6144/128) | 26 |
| FP32 | **31 TFLOPS** | 7.987 TFLOPS |
| FP4 (tensor) | 1 000 TOPS (1 PFLOP) | — |
| **FP64** | **NOT PUBLISHED — the key unknown** | 0.1248 TFLOPS (1:64) |
| L2 cache | **24 MB** | 6 MB |
| L4 / system cache | 16 MB (CPU L4) | — |
| memory | **128 GB LPDDR5x, unified coherent (UMA)** | 12 GB GDDR6 (discrete) |
| memory bandwidth | **273 GB/s** (marketing) / ~301 GB/s raw | 288 GB/s |
| memory interface | 256‑bit | 192‑bit |
| CPU | **20 Arm (10× X925 + 10× A725)**, 16 MB L3/cluster | — (host is an 8‑core x86) |
| CPU↔GPU fabric | **NVLink‑C2C coherent, 5× PCIe Gen5** | PCIe Gen4 |
| TDP | 140 W (GB10) / 240 W (system) | 70 W |
| pairing | 2× Spark via ConnectX‑7 (200 Gb/s) | — |

The two numbers that drive the forecast are **not** the 6144 cores or the 31 TFLOPS FP32.
They are (a) the **128 GB unified coherent memory** and (b) the **NVLink‑C2C zero‑copy**
fabric — and the one unknown is the **FP64 rate**.

---

## 2. Factor‑by‑factor forecast, tied to `emd_par.py`

### 2.1 Per‑block time: essentially unchanged (forecast)

The per‑block wall time is set by the thread‑0 serial fraction (extrema scan O(N) + Thomas
O(n)), which is one thread doing O(N) work. The GB10's 1.85× core count does not help a
single thread. The O(N) parallel phases (spline eval, subtract, SD) are hidden behind the
serial fraction, so the GB10's extra throughput on those phases is not exposed.

**Forecast:** T_blk(GB10, N=441k) ≈ T_blk(A2000, N=441k) ≈ **~9 s**, modulo a modest
improvement if the GB10's FP64 rate is better than 1:64 *and* any part of the per‑block work
is FLOP‑bound rather than latency‑bound. The extrema scan and Thomas solve are
**latency‑bound** (a sequential dependency chain of memory loads), not FLOP‑bound, so even a
3.8× FP64 rate (the 1:64 scenario) does little for them. **Expect ~9 s per block, same as
measured.**

### 2.2 Concurrency / the 128 GB win: the dominant effect (forecast)

On the A2000, 12 GB caps the resident blocks: at N=441k each block holds ≈21 MB (signal +
working arrays + output IMFs), so O(10²) blocks fit before the 12 GB is consumed; beyond
that, `emd_par.py` batches (and the wall time grows in waves, essay §5.2). On the GB10,
**128 GB removes the cap almost entirely**: 1 024 blocks at N=441k need ≈21 GB of working
set, trivially resident.

Because blocks run in waves and the per‑block time is ~9 s, the GB10 runs
**E = 1 024 in ~9 s (one wave)**, versus the A2000's E = 64 in ~22 s (2.4 waves). That is
~16× the ensemble size for a *shorter* wall time, i.e. ~4× better mode‑mixing reduction
(O(E^−1/2): √(1024/64) = 4). The `batch` auto‑sizing in `emd_par.py` (which reads free
device memory) will simply set `batch = E` and run one wave — no batching, no queueing.

This is the single largest GB10 advantage for EEMD: **the ensemble axis, which is the only
axis that actually pays (essay §3.3), becomes almost free in wall time and unbounded in E.**
The A2000 forced a trade‑off between E and memory; the GB10 removes it.

### 2.3 Memory bandwidth: no win for the O(N) phases (spec)

GB10: 273 GB/s. A2000: 288 GB/s. **Essentially identical.** The O(N) streaming phases
(extrema read, spline‑eval read+write, subtract read+write, SD read) run at the same
bandwidth on both. There is no bandwidth‑driven speedup. (The GB10's 256‑bit vs 192‑bit
interface is offset by the lower LPDDR5x vs GDDR6 data rate; the net is a wash.)

### 2.4 The 24 MB L2: a modest, real win (spec + forecast)

GB10: 24 MB L2. A2000 (GA106): 6 MB L2. The per‑block working set at N=441k is
3.5 MB (signal) + 17.6 MB (10×K arrays, K=N/2) ≈ 21 MB, which **fits in the GB10's 24 MB L2**
but not in the A2000's 6 MB. Across the 24 sifts, the signal and working arrays are re‑read
repeatedly; on the GB10 they are largely **L2‑resident**, so the DRAM traffic for those
re‑reads is reduced. This shortens the O(N) parallel phases. But because those phases are
already hidden behind the thread‑0 serial fraction (§2.1), the wall‑time impact is **small**
— it matters only if the serial fraction is also parallelized (then the parallel phases
emerge and the L2 advantage is fully exposed).

### 2.5 FP64 rate: the critical unknown (unknown → two scenarios)

The GB10's FP64 rate is **not published** in the datasheet, Hot Chips, or ISSCC materials I
could find. This is the single most important number to verify on the hardware, because EMD
is FP64‑bound (the 2‑dominant spline solve and the SD accumulation need double precision;
FP32 would inject O(10^−7) errors that the discontinuous‑selection iteration amplifies into
knot flips, essay §1.1).

- **Scenario A — desktop/Ada pattern (1:64):** 31/64 ≈ **0.48 TFLOPS** FP64, **3.8× the
  A2000's** 0.125 TFLOPS. Speeds up FLOP‑bound phases (spline eval, SD accumulation) by up to
  3.8×, but those are hidden behind the serial fraction, so wall‑time impact is small
  (~9 s → ~8.5 s per block, if at all).
- **Scenario B — data‑center rate (1:2 or better):** 15.5+ TFLOPS FP64. A genuine 100× over
  the A2000. Still mostly hidden behind the serial fraction, but now the parallel phases are
  fast enough that the serial fraction is the *only* thing left — and the L2‑resident
  parallel phases (§2.4) become the new floor.

Either way, **the FP64 rate does not change the qualitative picture**: the per‑block time is
set by the thread‑0 serial fraction, and the GB10's wins are concurrency (§2.2) and transfer
(§2.6), not raw FP64. But it determines how much headroom there is *after* the serial
fraction is parallelized.

### 2.6 NVLink‑C2C zero‑copy: eliminates the transfer overhead (spec)

On the A2000, the module's host overhead — the **host→device transfer** of the $(E,N)$ noise
+ signal, plus host‑side accumulation — is measured at **≈2.5 s on top of a 9.9 s kernel at
$E=8$, $N=441{,}000$** (≈25%), scaling sublinearly in $E$. On the GB10, the CPU and GPU share
a **coherent 128 GB UMA** over NVLink‑C2C (5× PCIe Gen5 bandwidth). The `x` array and the
per‑trial noise **do not need to be copied** to device memory — they are already in the
shared coherent space. `cp.asarray` becomes a view, not a transfer.

**Forecast:** the module end‑to‑end time collapses toward the raw kernel time (the ≈2.5 s
host term at $E=8$ disappears). The gain is modest at small $E$ (25% of the kernel time) and
grows with $E$ (the $(E,N)$ transfer is the term that scales in $E$); it is a *fabric* win,
not a *compute* win, and it is independent of the per‑block time.

### 2.7 The 20 Arm cores: the CPU backend becomes competitive (spec + forecast)

The GB10's host is 20 Arm cores (10× Cortex‑X925 high‑performance + 10× A725), versus the
8‑core x86 on which the A2000 numbers were taken. The `method='cpu'` backend in
`emd_par.py` uses `ProcessPoolExecutor` with `n_workers = min((E-1)//2, cpu_count()-1)`. On
the GB10, `cpu_count()-1 = 19`, so **E = 20 trials run concurrently on CPU**. The X925 is a
~3.5 GHz high‑performance core (faster per‑core than a typical 8‑core x86 workstation core),
so the per‑trial CPU time is lower.

**Forecast:** the CPU backend on the GB10 is competitive with the GPU for E ≤ ~20, and the
streaming + dynamic‑scheduling design (which the review praised) is exactly what makes it
robust at 20 workers. For E > 20, the GPU (one wave, ~9 s, unbounded E via 128 GB) wins. The
`method='auto'` heuristic (GPU if available) is correct, but on the GB10 the CPU/GPU crossover
moves to a larger E than on the A2000.

### 2.8 Software: CC 12.0, CUDA ≥ 12.8, numba‑cuda, and the bootstrap (spec)

- **Compute capability 12.0 (sm_120)** is a Blackwell target first supported by **CUDA 12.8**.
  The DGX Spark ships with a CUDA 12.8+ toolkit, so the kernel will compile.
- **numba‑cuda:** the built‑in `numba.cuda` target is deprecated (development moved to the
  `numba-cuda` package, still exposed under `numba.cuda`). It supports CC ≥ 3.5; the open
  question is whether the installed numba‑cuda's NVVM (from the Spark's CUDA toolkit) can
  target **sm_120**. CUDA 13.x NVVM definitely can (sm_120 is a first‑class target since
  12.8), so with a current numba‑cuda the kernel should compile to sm_120. **Verify on the
  hardware** with a one‑line `numba.cuda.is_available()` + a trivial kernel launch.
- **The `_bootstrap_nvvm()` hack does not transfer.** On the A2000 (Windows) we preloaded
  `cudart.dll`/`nvvm.dll` by full path because the CUDA 13.1 install shipped versioned names
  with no bare‑name file. On the DGX Spark (Linux, NVIDIA's own DGX OS with a proper CUDA
  install), numba finds the CUDA libraries through its normal search path
  (`/usr/local/cuda` or `CUDA_HOME`). The bootstrap should be **removed** (or left as a
  no‑op) on the Spark; the `cuda_libs/` directory and the `ctypes.CDLL` preload are
  Windows‑specific and would be dead code.
- **The kernel factory** (N, K, MAXIMF, nthreads baked as compile‑time constants, cached per
  tuple) is unchanged and works on sm_120. The JIT cost (~13 s on the first call at N=441k on
  the A2000) recurs per distinct tuple on the Spark, but is amortized over repeated calls at
  the same N.

---

## 3. Synthesis: what you would actually observe on the DGX Spark

For the reference sweep (N = 441 000), forecast, warm (JIT compiled):

| E | A2000 (measured) | GB10 (forecast) | why |
|---|---|---|---|
| 1 (plain EMD) | 9.2 s (kernel) | **~9 s** | per‑block time unchanged (§2.1) |
| 8 | 10.0 s (≈1.1 wave) + 2.5 s host | **~9 s (module)** | zero‑copy removes the 2.5 s host term (§2.6) |
| 64 | 21.8 s (≈2.4 waves) | **~9 s (module)** | 1 wave on 128 GB (§2.2) |
| 128 | ~40 s (≈4.4 waves) | **~9 s** | 1 wave; 128 GB resident |
| 1 024 | batches (12 GB cap, many waves) | **~9 s** | **1 wave, ≈21 GB resident (§2.2)** |

Three observations:

1. **The per‑block time (~9 s) is unchanged.** The GB10's 1.85× cores and (probably) higher
   FP64 do not touch the thread‑0 serial fraction. This is the honest, hardware‑agnostic
   limit, and it is the same on every GPU until the serial fraction is parallelized.

2. **The host overhead disappears.** The ≈2.5 s H2D transfer + accumulation at $E=8$ (≈25%
   of the kernel time) is removed by NVLink‑C2C zero‑copy. The gain grows with $E$ because
   the $(E,N)$ transfer is the term that scales in $E$. This is a *fabric* win, not a
   *compute* win.

3. **E becomes unbounded for the same wall time.** The 128 GB memory runs E = 1 024 in one
   ~9 s wave, versus the A2000's 12 GB cap (which forces multiple waves / batching at
   $E=64$). That is ~16× the ensemble at a *shorter* wall time, ~4× better mode‑mixing
   reduction. The ensemble axis — the only axis that pays — is effectively free on the GB10.

**What the GB10 does NOT fix:** the top‑octave sampling limit (2.2 samples/cycle, essay §5.5)
and the per‑block thread‑0 serial fraction. The first is a representation boundary (no
hardware removes it). The second is the one thing that would make the *per‑block* time drop,
and it is an **algorithmic** fix, not a silicon one: parallelize the extrema scan (prefix‑sum)
and the spline solve (block Thomas with a seam of width ~30 knots, justified by the
(2−√3)^|i−j| Green's‑function decay, essay §2.3 — the seam error is O((2−√3)^30) ≈ 10^−18,
below double precision, so the partitioned solve is exact in practice). That is exactly the
"overlapping localized sub‑splines" the review hinted at, and it is where the GB10's 48 SMs
and 24 MB L2 would finally be exposed (the parallel phases emerge once the serial fraction is
gone). Until that fix, the GB10 is a better *ensemble machine* (large E, zero‑copy) but not a
faster *per‑trial machine* than the A2000.

---

## 4. What to verify on the hardware (turns the forecast into data)

1. **FP64 rate** — the critical unknown. `nvidia-smi -q` does not report it; run a small
   FP64 triad / DGEMM microbenchmark (or `cupy`'s `matmul` in float64 and divide by 2·M³·t).
   This decides Scenario A vs B in §2.5.
2. **numba‑cuda sm_120 support** — `from numba import cuda; cuda.is_available()`, then launch
   a trivial `@cuda.jit` kernel. Confirms the NVVM in the Spark's toolkit targets sm_120.
3. **Per‑block time at N=441k** — `gpu_eemd_full(x, E=1)` warm. Confirms the ~9 s forecast
   (§2.1) and isolates the thread‑0 serial fraction on the GB10.
4. **One‑wave concurrency at large E** — `gpu_eemd_full(x, E=1024)` warm (pass a 2‑D
   `(1024, N)` array). Confirms the 128 GB
   runs 1024 blocks in one ~9 s wave (§2.2) and measures the actual L2 hit rate (Nsight
   Compute `lts__t_sectors_op_read` vs DRAM sectors).
5. **Zero‑copy transfer** — time `cp.asarray(x)` for a 3.5 MB array. On the Spark it should
   be ~0 (a view); on the A2000 it is a real H2D copy. Confirms §2.6.

These five measurements would convert every **forecast** above into a **verified** number.

---

## 5. Bottom line

The GB10/DGX Spark is a **better EEMD ensemble machine** than the A2000, for two reasons that
are *fabric and memory* properties, not compute properties: **128 GB unified coherent memory**
makes the ensemble axis (the only axis that pays) unbounded in E for a constant ~9 s wall
time, and **NVLink‑C2C zero‑copy** removes the host→device transfer that added ≈2.5 s of
overhead on the A2000. It is **not** a faster per‑trial machine: the ~9 s per‑block time is set
by the thread‑0 serial fraction (one thread doing O(N) work), which no amount of cores, FP64
rate, or L2 changes. The top‑octave sampling limit (2.2 samples/cycle) is likewise untouched.
The one number to verify before any of this is trusted is the **FP64 rate**, which is not
published. The per‑block time is only reduced by an algorithmic fix — parallel extrema
(prefix‑sum) and a decay‑seamed block Thomas — and that is where the GB10's silicon would
finally matter.
