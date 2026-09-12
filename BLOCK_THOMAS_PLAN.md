# Block-Thomas + Parallel Scan — Implementation Plan (r6 follow-up)

**Status: EXECUTING (user-approved 2026-09-12). Phase 0 COMPLETE on both
machines (see §3.1); Phase 1–4 in progress per the gates below.**

**Trigger.** Review 6 (`EEMD4CUDA_v3_r6_Gem.md`) measured the per-block time
`T_block` (E=1, N=441000) on both machines:

| | RTX 4070 Laptop (8 GB) | GB10 Spark (128 GB unified) |
|---|---|---|
| T_block (r6 estimate) | 4.62 s | 13.60 s (≈3× slower) |
| T_block (Phase 0 measured) | 4.08 s (`dbg_serial_split_v0`) | **11.92 s** (`dbg_serial_split_v0`, 2026-09-12) |
| E=64→128 wall growth | 12.05→21.86 s (×1.81, VRAM paging) | 30.07→36.01 s (×1.20, no paging) |

The 3× per-trial gap is the **thread-0 serial fraction**: one CUDA thread
walks 441k samples for the extrema scans and does the O(n) Thomas solve, so
`T_block` is a benchmark of the GPU's *single-thread scalar latency*. The
GB10 (low-clocked SoC cores) is exactly where that hurts. The plan is to
remove the serial fraction by parallelizing (a) the extrema scans and (b) the
spline solve — the "Decay-Seamed Block Thomas" of essay §6, plus the
companion parallel scan the essay lists alongside it.

**Honest expectation — REVISED by Phase 0 (measured, not modeled).** The
in-kernel `clock64` breakdown (see Phase 0 below) shows the **extrema scans
are 78.8% of `T_block`**, the Thomas solve only 8.2%. So the parallel *scan*
is the dominant lever, and the block-Thomas is secondary. Projected `T_block`
win on the 4070 is **~15×** (4.08 s → ~0.27 s), i.e. *larger* than the
pre-measurement 4–6× estimate. The GB10 still wins at large E because of
memory, which this change does not affect (footprint is unchanged).

---

## 1. Why both changes are needed (the serial-fraction anatomy)

Current kernel (`emd_gpu.py`, post-review-3 line numbers). Per **sift**,
thread 0 alone executes:

| # | Work | Lines | Iters/sift | Parallel? |
|---|---|---|---|---|
| 1 | MAX knot scan (detect + collect) | 187–195 | N = 441k | → YES (scan) |
| 2 | UPPER: H build | 201–203 | nm ≈ 220k | → YES (elementwise) |
| 3 | UPPER: system fill (dd,ll,uu,rr) | 204–209 | nint ≈ 220k | → YES (elementwise) |
| 4 | UPPER: Thomas solve | 210 | 2·nint ≈ 441k | → YES (block Thomas) |
| 5 | UPPER: Mf fill | 211–213 | nm ≈ 220k | → YES (elementwise) |
| 6 | MIN knot scan | 219–227 | N = 441k | → YES (scan) |
| 7 | LOWER: H build + fill + Thomas + Mf | 233–245 | ≈ 1.1M | → YES (same as 2–5) |

Per **IMF** (once, not per sift): residual extrema count, lines 161–170, N = 441k → YES (count scan).

Already parallel (all 128 threads): spline eval (215–216, 249–250),
mean/subtract/SD reduction (248–275), hv init (178–179), IMF store / residual
update (286–288).

**Total serial ≈ 3.1M iterations/sift** (2.65M by the r6 script's tally),
and the essay's A2000/4070 data shows the parallel phases are *hidden behind*
this serial fraction — i.e. the serial work is ~80–90% of `T_block`.
Parallelizing **only the Thomas solve** (33% of the serial work) would cap the
win at ~1.5× (Amdahl). Parallelizing the **scans + fills + solve** removes
~all of it. That is why the plan ships the pair, not the solver alone.

Post-fix serial budget per sift ≈ 2 × (window-Thomas imbalance tail + a few
syncthreads) ≈ **< 20k iterations, mostly parallel** — a ~150–300× reduction
of the serial fraction. The new per-sift floor is the *already-parallel*
spline evaluation (2 × N samples × ~17 L2 reads per thread, binary search),
which this plan does not touch.

## 2. Algorithm choice

### 2.1 The spline solve: seam-window Thomas (Option A) — recommended

Partition the nint interior unknowns into contiguous blocks of **B = 250**
(owned). Each block solves its **window** = owned region ± **S = 30** halo
(wide, i.e. B+2S = 310 unknowns) as a *standalone* tridiagonal system — the
external couplings at the two window edges are dropped (equivalently, zero
boundary conditions at the window rims). The owned values are the answer; the
halo values are discarded (each halo element is solved redundantly by its
neighbour, never written twice — ownership is disjoint).

**Error budget (verified this session, not assumed):**

- The Green's function of the 2-dominant spline system decays as
  `ρ^|i-j|` with local base `r_i ≤ 2−√3 = 0.267949` for **all** spacings
  (AM-GM: `r_i + 1/r_i = 2(1+Hᵢ/Hᵢ₊₁) ≥ 4`). Uniform spacing is the *slowest*
  decay; measured bases: uniform 0.267949, CV≈0.4 → 0.2502, CV≈0.8 → 0.2269,
  CV≈1.5 → 0.2136, alternating 1/10 spacing → 0.1616.
- An owned point is ≥ S = 30 from both window rims, so the seam error is
  `O(ρ³⁰ · scale) ≤ 6.9e-18 · C · scale` — **10× below double-precision
  ε ≈ 2.2e-16**, and ~4 orders below the existing GPU-vs-CPU noise floor
  (1.4e-13, which itself is chaotic amplification of 1-ulp differences by
  ×10²–10³ over a full decomposition).
- Consequence: the gate threshold for full-decomposition agreement can stay at
  the existing A2 level (≤ 1e-8) with a *tight* internal gate of 5e-13.

**Why Option A over the alternatives:**

- **vs exact 1-level block Thomas (Option B, the "reduced system" variant):**
  Option B is algebraically exact (each block eliminates its B−2 interior
  unknowns, producing a tridiagonal reduced system of 2·nb ≈ 1764 boundary
  unknowns solved by a *serial* Thomas, then a parallel recovery pass). It is
  correct but ~2× the code (reduced-system assembly + recovery), needs the
  affine bookkeeping (A, β, γ per block), and still leaves a 1.7k-step serial
  phase. Option A buys the same speed with a *provably sub-ε* error and no
  serial phase. Option B is the **fallback** if §6 verification shows seam
  effects on pathological spacing (it shouldn't — the bound is spacing-
  independent — but the fallback is cheap to keep).
- **vs cyclic reduction / prefix-sum (essay §3.2's O(log n) option):**
  rejected in review 1 for good reason — O(n log n) work with a non-coalesced
  global-memory storm; the serial 2·nint Thomas is only 441k cheap L2-resident
  steps, and the seam variant parallelizes it with zero extra asymptotics.
- **vs warp shuffles (r3 P5):** numba has no `__shfl_down_sync`; unchanged.

### 2.2 The extrema scans: detect + shared-scan + compact (Option A)

Three scan sites (residual count per IMF; MAX and MIN knot collect per sift)
all reduce to: *detect 1-stencil extrema, count/compact in time order.*

Per site, two passes over the block's 128 threads (no extra global memory):

1. **Pass 1 (count):** `for i in range(tid+1, N-1, nthreads)`: detect
   `(hv[i] > hv[i-1]) && (hv[i] > hv[i+1])` → local count `c[tid]` in shared.
   Hillis-Steele inclusive scan over the 128 counts (7 `syncthreads`);
   `total = v[127]` → `nm_s[0]` (the true count, exactly as today);
   exclusive offsets `off[tid] = v[tid-1]`.
2. **Pass 2 (compact):** re-detect in the same strided order, writing
   `kt[off + local_idx] = i`, `ky[off + local_idx] = hv[i]` (capped at K,
   same guard as today). Re-detection is 2 comparisons — negligible.

**Determinism:** writes happen in strictly ascending `i` (per-thread stride
ascending, thread offsets from the scan) → `kt`/`ky` are **bit-identical** to
the current single-thread scan. The count is exact. The `K` cap (K = N//2 ≥
max possible extrema = ceil((N-2)/2)) can never fire — same as today.

The H build (`H[j] = kt[j+1]-kt[j]`), system fill (dd/ll/uu/rr), and Mf fill
become plain strided elementwise loops — trivially parallel, no scans.

### 2.3 Parameters

| param | value | rationale |
|---|---|---|
| B (block) | 250 | essay §6; nb = ceil(nint/250) = 882 blocks → 6.9 windows/thread at 128 threads; window = 310 → ~4.4k steps/thread/envelope |
| S (halo) | 30 | ρ³⁰ = 6.9e-18 < ε; S=20 would give 3.6e-12 — inside the gate but no margin; keep 30 |
| nthreads | 128 | unchanged |

Sensitivity study (B ∈ {128, 250, 512}) is a Phase-4 benchmark item, not a
correctness item.

## 3. Phase 0 — Measure before building (read-only; throwaway code)

1. **Serial/parallel split on the 4070.** Build a *throwaway* kernel variant
   (separate file, e.g. `dbg_serial_split.py`) with the parallel eval/SD
   phases stubbed out (compile-time flag), measure pure-serial `T_block`;
   compare to the full 4.62 s. Confirms the Amdahl model (expected: serial
   ≈ 80–90%). Discard afterwards.
2. **GB10 FP64 rate** (essay §6: "the first thing to measure"): a cupy
   float64 matmul microbenchmark script (e.g. 4096³, 5 repeats) — run by the
   user on the Spark. Sets the post-fix headroom ceiling.
3. Re-capture the **pre-change GPU baseline** (reuse `capture_baseline_r3.py`
   → `baseline_r3_preBT.npz`; the existing `baseline_r3.npz` was verified
   bit-identical to the current kernel after P4, so it may be reused as-is —
   confirm with one A/B run first).

### 3.1 Phase 0 RESULTS (completed 2026-09-11, 4070)

Method: `dbg_serial_split_v0.py` — a faithful copy of the production kernel
with a PTX-injected `clock64()` (numba 0.67 exposes no device clock; the
`@intrinsic` + `mov.u64 $0, %clock64;` trick from review 2 works) timestamping
each phase. **The breakdown closes to wall time at 100%** (phase-sum ÷ 4.079 s
wall = 2.37 GHz = the 4070-Laptop boost clock), so it is the real split, not a
model. `phase0_microbench.py` supplies the hardware rates (run again on the
GB10 for the side-by-side).

**Measured `T_block` split (E=1, N=441000, 24 sifths, 12 IMFs):**

| phase | % of T_block | real time | tag |
|---|---|---|---|
| MAX scan (1 thr) | 30.6% | 1.25 s | serial |
| MIN scan (1 thr) | 30.3% | 1.24 s | serial |
| resid_count (1 thr, per-IMF) | 17.8% | 0.73 s | serial |
| UPPER build | 4.2% | 0.17 s | serial |
| LOWER build | 4.2% | 0.17 s | serial |
| UPPER thomas | 4.1% | 0.17 s | serial |
| LOWER thomas | 4.1% | 0.17 s | serial |
| **extrema scans (MAX+MIN+resid) subtotal** | **78.8%** | **3.21 s** | serial |
| PAR eval+SD (128 thr) | 3.9% | 0.16 s | parallel |
| UPPER/LOWER Mf fill | 0.6% | 0.02 s | serial |
| PAR store (128 thr) | 0.3% | 0.01 s | parallel |
| **SERIAL total** | **95.8%** | **3.91 s** | |
| **PARALLEL total (existing floor)** | **4.2%** | **0.17 s** | |

**Key findings:**
1. **The extrema scans are the bottleneck (78.8%), not the Thomas (8.2%).**
   The r6 review correctly diagnosed "single-thread scalar latency" but
   attributed it to Thomas; the scan dominates 10:1. The parallel-scan is the
   primary lever. Per-iter costs: scan 288 cyc, Thomas 789 cyc, build 801 cyc —
   Thomas is 2.7× costlier *per iter* but 4× less *work*.
2. **Scan parallelism is very high.** `phase0_microbench.py`: the production
   extrema pattern (3 loads/iter) runs **221 cycles/elem on 1 thread vs
   2.89 on 128 threads → 72.6×**. This is the single biggest available win.
3. **FP64 rate (4070): 0.257 TFLOPS** (2:64-ish, as expected for Ada). The
   GB10's FP64 rate is the open unknown (run `phase0_microbench.py` on the
   Spark) — it sets the post-fix ceiling for the eval phase, which becomes the
   new floor.
4. **Single-thread scalar rate (4070): 59.5 cycles/iter** on a dependent
   FLOP chain — confirms the "single-thread trap."
5. **Baseline A/B: CONFIRMED** — the current post-P4 kernel reproduces
   `baseline_r3.npz` at max|d| = 0.0 (both configs, nsift/nmode equal). The
   Phase-3 bit-exact gate is ready.

**Revised projection (4070):** applying the measured speedups — scan /72.6,
build /70, block-Thomas /6.9 (882 blocks / 128 thr), Mf /70, plus the existing
4.2% parallel floor — gives **new `T_block` ≈ 0.27 s, a ~15× speedup** (range
~0.27–0.40 s with sync/launch overhead). The new floor is the already-parallel
eval+SD (0.17 s), which this plan does not touch.

**GB10 side-by-side — MEASURED (2026-09-12, `GB10-Numberz.txt`; reviewed in
`dbst_v1_r3_Gem.md`):**

```text
device: NVIDIA GB10 SMs=48  L2=24.0 MB  clock=2418 MHz
FP64 matmul 4096x4096: 1.042 s -> 0.132 TFLOPS
scalar dependent FLOP chain (1 thread, 4M iters): 64.28 cycles/iter
stream scan N=441000: 1 thread 224.7 cycles/elem; 128 thr 4.43 cycles/elem
                     -> 50.1x scan speedup
T_block (dbg_serial_split_v0, E=1, N=441000, 24 sifths, 12 IMFs):
  wall = 11.919 s
  MAX scan 3042 ms (27.9%) | MIN scan 3010 ms (27.6%) | resid_count 2167 ms (19.9%)
  LOWER build 540 ms (5.0%) | UPPER build 538 ms (4.9%)
  LOWER thomas 521 ms (4.8%) | UPPER thomas 521 ms (4.8%)
  PAR eval+SD 462 ms (4.2%) | PAR store 30 ms (0.3%)
  UPPER/LOWER Mf fill 29 ms each (0.3% each)
  SERIAL total 10398 ms = 95.5% | PARALLEL total 492 ms = 4.5%
```

**GB10 findings (r3 review sign-off):**

1. **The FP64 mystery is solved — hard cap.** 0.132 TFLOPS realized = the
   consumer/Ada-class 1:64 FP64 segmentation (31 TFLOPS FP32 vs ~0.48
   TFLOPS theoretical FP64). The already-parallel `PAR eval+SD` phase takes
   **461 ms** on the GB10 and is the post-fix floor: without float32 eval
   (a separate future plan) `T_block` can never go below ~0.5 s on this chip.
2. **The bottleneck is identical across architectures:** serial fraction
   95.5% (GB10) vs 95.8% (4070); scans 75.4% vs 78.8%. 128 GB of unified
   memory does not outrun the single-thread Amdahl trap; the scalar chain
   rate (64.28 vs 59.5 cycles/iter) is essentially the same silicon physics.
3. **Revised GB10 projection (exact math from the measured phases):**
   scans 8.21 s × (1/50.1) → **0.16 s**; build+Thomas+Mf 2.16 s ×
   (128/882 ≈ 1/6.9) → **0.31 s**; existing parallel floor **0.49 s**
   → **T_block ≈ 0.96 s, call it 1.0–1.5 s** with sync/launch overhead.
   That is **11.92 s → ~1.2 s ≈ 10×**, and it is what lets the GB10 finally
   use its 128 GB pool for extreme ensembles (E=1024) without minutes.
   (The pre-measurement "2–3.5 s" estimate was conservative; the FP64 floor
   and the 50.1× scan number tighten it.)

**Phase 0 status: COMPLETE on both machines.**

## 4. Phase 1 — Host-side reference + unit verification (new file, no kernel)

New file `test_block_thomas.py` (host-only parts first):

1. **Seam-window solver in numpy** (mirror of the device algorithm exactly:
   same B, S, same clipped windows, same dropped edge couplings).
2. **Gate U1 (solver vs exact):** for knot sets from (a) the reference-sweep
   EMD (real knots, upper & lower), (b) uniform grid, (c) non-uniform CV≈0.4
   (the r6 script's case), (d) pathological alternating 1/10 spacing,
   (e) random CV≈1.5:
   `max|ΔM| / max|M| ≤ 1e-15` vs `emd_ref.thomas` (exact) and vs
   `np.linalg.solve` (independent). Expected: ~1e-17 (seam error) + rounding.
   **Fail → do not proceed to the kernel; go to Option B (exact block
   Thomas) or increase S.**
3. **Gate U2 (scan equivalence, host emulation):** emulate the two-pass
   scan on random signals (including flat runs, plateau extrema, single
   extremum) and assert `kt`/`ky`/count bit-identical to a serial scan.
4. **Gate U3 (decay-base re-check in-file):** assert measured base ≤
   2−√3 on cases (a)–(e) (guards the whole error argument against drift).

## 5. Phase 2 — Kernel implementation (`emd_gpu.py` only)

Files touched: **`emd_gpu.py`** (kernel + helpers + factory) and the new
**`test_block_thomas.py`**. Explicitly **untouched**: `emd_par.py`,
`emd_ref.py`, `test_emd_gpu.py`, `test_review3_fixes.py`,
`final_module_test.py`, `capture_baseline_r3.py`.

1. **Device helpers** (new, `@cuda.jit(device=True, inline=True)`):
   - `_window_thomas(dd, ll, uu, rr, xx, cc, w0, w1)`: plain Thomas over the
     window `[w0..w1]` with the edge couplings dropped (row w0 uses d only,
     row w1 uses d only). Writes multipliers into `cc[w0..w1]`, solution into
     `xx[w0..w1]` — **in place, reusing the existing W slabs; no new global
     buffers.**
   - `_scan_extrema(arr, kt, ky, cnt_sh, off_sh, N, K, nthreads, want_max)`:
     the two-pass detect/scan/compact of §2.2 (shared arrays passed in;
     returns total).
2. **Kernel body changes** (inside `_make_full_kernel`):
   - Lines 161–170 (per-IMF count): → `_scan_extrema(rs, ..., want_max)` +
     `want_min`, counts only.
   - Lines 187–195 (MAX collect): → `_scan_extrema(hv, kt, ky, ..., want_max)`.
   - Lines 201–213 (UPPER build+Thomas): → strided H build; strided system
     fill; `for blk in range(block_id, nb, nthreads): _window_thomas(...)`;
     strided owned-copy `Mf[i+1] = xx[i]`.
   - Lines 219–245 (MIN + LOWER): → same pattern.
   - All other logic (stop flag, SD reduction, IMF store, residual update)
     **unchanged**.
3. **Kernel factory:** new compile-time consts `B`, `S` (and an `EXACT` bool,
   default False, that selects the old serial `_thomas` for in-kernel A/B
   debugging — kept out of the production path). `_KERNEL_CACHE` key gains
   `(B, S, EXACT)`. `_pick_K` unchanged.
4. **Storage audit (no new global memory):** W slabs reused (dd, ll, uu, rr,
   xx, cc for the window Thomas; H, ky, Mf, kt as today). Shared memory: +2 ×
   128 × int32 (cnt/off) ≈ 1 KB → total ≈ 2 KB, far below the 48 KB limit.
   Device footprint per trial: **unchanged** (B4/footprint numbers in the
   benchmark stay valid).
5. **numba caveats** (from the P4 experience): every int32 knot-index use
   needs explicit casts (the scan writes `np.int32(i)`; H build
   `float(kt[j+1]-kt[j])`); the Hillis-Steele scan needs `cuda.syncthreads()`
   after each width step; window loops use clipped `w0/w1` (first/last block
   windows touch 0 and nint-1).

## 6. Phase 3 — Integration verification (gates)

Against the pre-change baseline (`baseline_r3.npz`, configs A: E=1 N=110250;
B: E=8 seed-7 N=110250) and the CPU reference:

| Gate | Check | Threshold | Rationale |
|---|---|---|---|
| G1 | per-IMF max\|d\| vs pre-change GPU baseline, configs A & B | ≤ 5e-13 | seam error 6.9e-18·scale amplified ×10²–10³ by the chaotic sifting; 3× margin over the observed 1.44e-13 CPU-vs-GPU level |
| G2 | nmode agreement (GPU new vs pre-change & vs CPU) | equal, ±1 reported and must be explained | a τ-boundary stop flip is possible in principle at the 1e-15 level |
| G3 | per-trial total_sifts | \|diff\| ≤ 1 | same reason |
| G4 | partition of unity (recon) | ≤ 1e-14 | unchanged by construction (residual = x − Σimfs) |
| G5 | determinism (A3-style, same seed twice) | bit-identical (max\|d\| = 0) | window Thomas and the scan are deterministic; must stay so |
| G6 | full `test_emd_gpu.py` battery | A1–A5 pass (A2 threshold 1e-8) | existing acceptance |
| G7 | `test_review3_fixes.py` + `final_module_test.py` | pass | regression (CPU path untouched; GPU timings reported, not gated) |
| G8 | pathological-spacing E2E: a signal whose knot spacing CV ≈ 1.5 (e.g. a strongly modulated chirp) through `decompose(method='gpu')` vs CPU | per-IMF ≤ 1e-12 | the seam-error stress case end-to-end |

**Any G1–G5 failure → stop, diagnose, and fall back to Option B (exact
block Thomas, §2.1) before touching anything else.**

## 7. Phase 4 — Benchmark & tuning

1. Part B of `test_emd_gpu.py` on the 4070 before/after (T_block, C_est
   sweep, B4 host overhead). Expected: T_block 4.62 s → **~0.8–1.5 s**
   (new floor = the parallel spline eval); C_est at fixed E rises
   proportionally.
2. Same on the GB10 (user runs): T_block 11.92 s (measured) → **~1.0–1.5 s**
   expected (r3 review: 0.16 s scans + 0.31 s build/Thomas/Mf + 0.49 s FP64
   eval floor ≈ 0.96 s); the E=64→128 growth should stay ~×1.2 (memory,
   unchanged). The 0.46 s FP64 eval phase is the hard floor on this chip
   (1:64 FP64 segmentation, 0.132 TFLOPS measured).
3. B/S sensitivity (B ∈ {128, 250, 512}, S = 30): pick the B with the best
   T_block; S is fixed at 30 by the error budget.
4. Update the essay's claim table (§7) with measured numbers when they land.

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Seam error larger than budget on pathological spacing | very low (bound is spacing-independent, verified to 1/10 spacing) | G1/G8 gates; Option B fallback; S 30→40 (ρ⁴⁰ = 1.3e-23) |
| numba codegen issue in the shared scan (syncthreads placement) | low-medium | host emulation (U2) first; EXACT flag for in-kernel A/B |
| Window-Thomas load imbalance (882 blocks / 128 threads) | certain but bounded (≤1 window ≈ 15% tail) | measured in Phase 4; static balanced assignment if it matters |
| Chaotic amplification flips a stop decision (G2/G3 ±1) | low | thresholds allow ±1 with mandatory explanation; if unexplained, Option B |
| The parallel eval becomes the new floor sooner than expected (T_block only 2–3×) | medium | that's the honest ceiling; Phase 0 measurement sizes it; the eval itself (wider L2 use, float32 eval) is a *separate* future plan |
| GB10 numbers don't match the 4–6× projection | resolved (Phase 0 measured 11.92 s; revised projection 1.0–1.5 s, §3.1/§7) | Phase 4 on-metal confirms; the plan's promise is "remove the serial fraction", the silicon sets what remains |

## 9. Acceptance (definition of done)

1. G1–G8 all pass on the 4070; G6–G7 green.
2. T_block on the 4070 ≤ 1.5 s at N=441000 (≥ 3× over 4.62 s).
3. GB10 on-metal run (user): T_block ≤ 4 s, E-scaling unchanged.
4. `emd_par.py`/`emd_ref.py`/existing tests byte-identical (diff = 0).
5. Plan document updated with measured Phase 0/4 numbers.

## 10. Effort estimate

- Phase 0: ~1 h (throwaway kernel + microbench scripts)
- Phase 1: ~2 h (numpy reference + unit gates)
- Phase 2: ~4–6 h (kernel surgery; the P4 bit-exact experience de-risks the numba typing)
- Phase 3: ~1 h wall (gates are fast; the 441k E2E G8 is the long one)
- Phase 4: ~1 h on the 4070 + user time on the GB10

**Total: ~1.5–2 days of work, gated at every phase boundary.**
