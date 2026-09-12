# Decay-Seamed Block Thomas v0 - review 1

##

As a **quick sanity-check script**, it's useful. As a **serious tester/validator of a Decay-Seamed Block Thomas (DSBT) algorithm**, it's quite limited.

### What it does well

The script checks three ideas:

1. **Theoretical decay constant**

   * Computes \(\rho = 2-\sqrt{3}\approx 0.267949\).
   * Prints \(\rho^s\) for several seam widths.
   * This is relevant because DSBT relies on exponential decay across block boundaries. 

2. **Empirical inverse decay**

   * Builds a natural-spline tridiagonal system on a nonuniform knot sequence.
   * Inverts the matrix and measures decay of a central column of \(A^{-1}\).
   * Fits a log-slope to estimate an empirical decay factor. 

3. **Back-of-the-envelope parallelism model**

   * Estimates number of blocks, seam overhead, and thread work.
   * Gives a rough speedup intuition. 

Those are all reasonable checks for the *motivation* behind a decay-seamed method.

---

## What it does **not** test

The major issue is that it never actually runs a Decay-Seamed Block Thomas solver.

The script:

* builds a matrix,
* computes a full inverse,
* measures decay,
* estimates workload,

but nowhere does it:

* partition the system into blocks,
* apply seam halos,
* solve local block systems,
* stitch blocks,
* compare DSBT output against exact Thomas output.

So it validates an assumption of DSBT (inverse decay), not the algorithm itself.

I would call it:

> **a DSBT feasibility demonstration, not a DSBT correctness test.**

---

## Missing correctness tests

A real DSBT tester should verify:

### 1. Solution accuracy

Generate many tridiagonal systems:

$$
Ax=b
$$

and compare:

* Exact Thomas solution \(x_T\)
* DSBT solution \(x_D\)

using

$$
\|x_D-x_T\|_\infty
$$

and

$$
\frac{\|x_D-x_T\|_2}{\|x_T\|_2}
$$

for varying seam widths \(s\).

---

### 2. Error-vs-seam-width curve

The core DSBT prediction is

$$
\text{error}\sim \rho^s .
$$

You should test:

```text
s = 5
s = 10
s = 15
s = 20
s = 25
s = 30
```

and verify that the measured error decreases exponentially.

That is arguably the single most important validation.

---

### 3. Boundary-seam error localization

DSBT predicts that most error is concentrated near seams.

Plot

$$
|x_D - x_T|
$$

across the domain.

You should see spikes near block interfaces and negligible error elsewhere.

---

### 4. Nonuniform-grid stress tests

The current script only uses one coefficient variation level:

```python
CV ~ 0.39
```

from the random spacing generation. 

A stronger tester would vary:

```text
CV = 0.0
CV = 0.1
CV = 0.3
CV = 0.5
CV = 1.0
```

and verify that decay and accuracy remain acceptable.

---

### 5. Pathological matrices

Test:

* nearly singular systems,
* highly varying diagonal dominance,
* alternating coefficients,
* clustered knots,
* random coefficient distributions.

These often reveal seam-instability issues.

---

### 6. Runtime measurement

The script estimates work analytically. 

A proper tester should measure:

```python
time_thomas
time_dsbt
speedup = time_thomas / time_dsbt
```

for increasing problem sizes.

Without timings, performance claims remain theoretical.

---

## One methodological concern

This section:

```python
G = np.linalg.inv(A)
```

computes the entire inverse. 

For \(n=4096\) that's fine, but for validating DSBT it is somewhat indirect.

A stronger decay test would estimate selected inverse-column entries through solves rather than constructing the full dense inverse. That scales better and resembles how DSBT would actually be used.

---

## Overall assessment

If the goal is:

* "Does the decay assumption behind DSBT appear valid?"

then the script is **good (about 7/10)**.

If the goal is:

* "Does my Decay-Seamed Block Thomas implementation work correctly and achieve the predicted error/runtime behavior?"

then the script is **poor (about 3/10)** because it never exercises the algorithm itself.

The biggest missing component is a direct **DSBT-vs-Thomas accuracy benchmark across seam widths and block sizes**. That would turn this from a conceptual validation script into a genuine algorithm test harness.
