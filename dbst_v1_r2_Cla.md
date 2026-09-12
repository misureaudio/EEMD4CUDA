# Decay-Seamed Block Thomas v1 - review 2

##

Overall, this is a solid proof-of-concept validation script — it correctly tests the central theoretical claim of DSBT (that error decays exponentially with seam width) and includes one clever sanity check. But it falls short of being a rigorous test suite in several ways. Here's the breakdown:

## What it does well

- **Tests the right hypothesis.** The core claim of DSBT is presumably "error decays like ρˢ with seam width," and Test 1 directly measures this against theory across s = 5–30. That's the correct experiment to validate the algorithm's premise.
- **Localization check is a nice touch.** Verifying that the max error occurs exactly at a block boundary (`dist_to_boundary == 0`) confirms the error mechanism is what you'd expect (boundary-seam truncation), not some other bug.
- **CV stress test** checks robustness to non-uniform grids, which matters since the ρ = 2−√3 decay rate is only exact for the *uniform*-spacing case (it's the known root for the constant 1,4,1 tridiagonal pattern from natural cubic splines).
- **Uses `thomas_solve` itself as ground truth**, which is reasonable since it's a textbook-exact algorithm — but see below.

## Notable gaps

**1. Performance is never measured.** `import time` is present but literally unused. Since the entire point of a block method like DSBT is presumably parallelizability/speed (each block can be solved independently), the complete absence of any timing or scaling benchmark is a real hole — the script only validates *accuracy*, not the thing that would justify using DSBT over plain Thomas in the first place.

**2. No independent validation of the "exact" reference.** `thomas_solve` is trusted as ground truth but never itself checked against, e.g., `np.linalg.solve` or `scipy.linalg.solve_banded` on a small case. If there's a subtle bug in `thomas_solve`, every downstream comparison is silently wrong.

**3. Single random seed, single N.** Everything uses `rng.default_rng(42)` and `N=100000` — one realization of one problem size. There's no statistical spread (mean/max error over multiple seeds) and no test of how error/behavior scales with N or with block count.

**4. Block size `b` is never varied.** Both tests fix `b=250`. The interaction between `s` and `b` (e.g., what happens as `s` approaches `b/2`, or `s > b`) is untested — an important edge case for a seam-based method.

**5. No edge-case coverage for the block partition.** N=100000 is exactly divisible by b=250, so the "last partial block" code path (`end = min((i+1)*b, N)`) never gets exercised with a genuinely partial final block. Also, boundary blocks at the very start/end of the domain (i=0 and i=nblocks-1) have asymmetric or missing seams by construction (`pad_start=max(0,...)`) — their error behavior is never isolated or reported separately from interior blocks.

**6. No assertions — it's a printout, not a test.** There are no `assert`/pass-fail checks; a human has to eyeball the tables to judge success. This makes it a demonstration script rather than something you could run in CI to catch regressions.

**7. The theory comparison is only strictly valid at CV=0.** ρ=2−√3 is the exact decay rate for the *uniform*-spacing spline matrix. Test 1 runs it at CV=0.4 and shows it alongside theory as if they're directly comparable — that's fine as a rough sanity check, but it's not rigorous, and the script doesn't clarify this limitation or test whether the *empirical* decay rate at various CVs tracks ρ_local (a locally-varying rate) instead.

**8. Only L∞ error is reported** — no relative error (normalized by ‖x_exact‖) and no L2/RMS error, so you don't know if the error is a global systematic bias or a rare/localized spike (the localization check partially compensates for this, but only at s=15).

## Bottom line

As a first-pass sanity check / proof-of-concept demo for a paper or writeup, it's reasonable — it tests the right qualitative claim and includes a good diagnostic (boundary localization). As a rigorous *test suite* you'd want to trust for correctness or ship with the algorithm, it's missing: ground-truth cross-validation, performance benchmarking (despite importing `time`), multi-seed statistics, variation of `b` and `N`, edge cases in block partitioning, and automated pass/fail assertions.