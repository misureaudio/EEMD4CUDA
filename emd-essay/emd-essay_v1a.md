# Empirical Mode Decomposition, Ensemble EMD, and CEEMDAN:
## A Nonlinear Multiresolution Analysis with Stochastic Regularization

*Prepared for an audience of mathematical physicists and analysts working in functional and numerical analysis with a focus on nonlinear methods.*

---

## 1. Introduction

The decomposition of a time series into oscillatory components is, at bottom, a problem in multiresolution analysis. The classical answers — the Fourier transform, the short-time Fourier transform, and wavelet expansions — all rest on a *fixed*, *linear*, and *shift-invariant* basis: a family of global sinusoids, or of sinusoids modulated by a fixed window, or of dilates and translates of a single mother function. These bases are optimal for signals whose oscillatory content is harmonic, stationary, or self-similar across scales, respectively. They are not optimal for the broad class of signals that arise in fluid turbulence, geophysics, cardiac and neural dynamics, or the output of nonlinear dynamical systems, where the instantaneous amplitude and frequency vary in a way that is neither periodic nor self-similar and that no single fixed basis captures without a loss of local information.

Empirical Mode Decomposition (EMD) was introduced by Huang, Shen, Long, Wu, Shih, Zheng, Yen, Tung, and Liu in 1998 [1] precisely to address this gap. Its defining property is that it prescribes **no basis at all**: the oscillatory components — the *intrinsic mode functions* (IMFs) — are extracted adaptively from the data by a purely local, nonlinear sifting procedure. The motivation is the Hilbert–Huang transform (HHT): given an IMF, the Hilbert transform yields a well-defined instantaneous frequency, so a decomposition into "well-behaved" oscillations is the prerequisite for a meaningful time–frequency representation of a nonlinear, nonstationary signal.

The price of this data-adaptivity is that the method is a **nonlinear operator** on the signal space, and nonlinear operators do not come with the convergence, stability, and uniqueness guarantees that linear spectral methods enjoy. As originally formulated, EMD has no general existence or convergence theorem, its stopping rule is ad hoc, and it exhibits well-known pathologies — most notably *mode mixing* — that limit its reliability. Two subsequent refinements, **Ensemble EMD (EEMD)** [2] and **Complete EEMD with Adaptive Noise (CEEMDAN)** [3], address these pathologies by a *stochastic regularization* of the nonlinear operator: one perturbs the signal with white noise, decomposes each perturbed copy, and averages. This essay develops EMD, EEMD, and CEEMDAN from a functional- and numerical-analysis standpoint, separating carefully what is rigorously established (the convergence of the *linear* iterative-filtering idealization of sifting, and the Monte-Carlo error scaling of the ensemble methods) from what remains heuristic (the convergence of the fully nonlinear spline sifting, the optimality of the stopping rule).

A single **running example** is carried through the whole essay: a two-tone amplitude- and frequency-modulated signal

$$
x(t) \;=\; a_1(t)\,\cos\big(\varphi_1(t)\big) \;+\; a_2(t)\,\cos\big(\varphi_2(t)\big),
\tag{1}
$$

where the two instantaneous frequencies $\varphi_1'(t)$ and $\varphi_2'(t)$ are *close* over a sub-interval $I\subset[0,T]$ (a frequency-overlap region) and *separated* elsewhere, and where $a_1$ and $a_2$ vary slowly. This is the minimal signal on which the distinction between the three methods becomes visible: it is simple enough to reason about, and rich enough to exhibit mode mixing, which is the central pathology that EEMD and CEEMDAN are designed to cure.

---

## 2. The EMD algorithm

### 2.1 The intrinsic mode function

An **intrinsic mode function** (IMF) is a function $h(t)$ satisfying two conditions [1]:

1. *(counting)* the number of local extrema and the number of zero crossings in the whole record are equal, or differ by at most one;
2. *(local zero mean)* the mean of the values defined by the upper envelope (the spline through the local maxima) and the lower envelope (the spline through the local minima) is zero, i.e. the two envelopes are symmetric about the local mean.

Condition (1) is a *local* AM–FM regularity requirement: it forces $h$ to be a "single" oscillation at every local time scale, so that an instantaneous frequency $\omega_h(t)=\varphi_h'(t)$ defined by the zero-crossing rate is meaningful. Condition (2) is the *zero-mean* requirement that makes the Hilbert transform of $h$ well-behaved (it removes the DC offset and, to first order, the slow amplitude trend). In the language of the HHT, an IMF is a function of the form $a(t)\cos\varphi(t)$ whose amplitude $a$ and phase $\varphi$ are slowly varying relative to the oscillation, so that one can speak of a local amplitude and a local frequency.

Note that the IMF is **not defined by a linear constraint** (it is not the range of a projection, and it is not a member of a fixed subspace). It is defined by a *data-dependent* combinatorial and local-symmetry condition. This is the first point at which EMD departs from every linear spectral method, and it is the source of both its power and its analytical difficulty.

### 2.2 The sifting process

Given a signal $x(t)$, EMD extracts the *fastest* oscillation by an iterative **sifting** process [1]. Starting from the current residual $r^{(0)}=x$:

1. **Locate** all local maxima and minima of $r^{(k)}$.
2. **Interpolate** the upper envelope $E_+(r^{(k)})$ by a spline through the maxima, and the lower envelope $E_-(r^{(k)})$ by a spline through the minima.
3. **Subtract the mean envelope**:
$$
r^{(k+1)} \;=\; r^{(k)} \;-\; \tfrac{1}{2}\Big(E_+(r^{(k)})+E_-(r^{(k)})\Big).
\tag{2}
$$
4. **Repeat** steps 1–3 until a stopping criterion (Section 2.3) is met; the output of the sifting is the first IMF, $C_1$.
5. **Subtract** $C_1$ from $x$ to form the first residual $R_1 = x - C_1$, and **recurse**: apply the same sifting to $R_1$ to obtain $C_2$, and so on.

The process terminates when the residual $R_K$ is monotone (or has fewer than two extrema), from which no further IMF can be extracted. The result is the decomposition

$$
x(t) \;=\; \sum_{k=1}^{K} C_k(t) \;+\; R_K(t),
\tag{3}
$$

with $C_1$ the fastest mode, $C_K$ the slowest, and $R_K$ the trend.

In **running example (1)**, the first sifting step is meant to isolate the fastest of the two tones. Where the two instantaneous frequencies are well separated, this works as intended: $C_1$ tracks the faster tone $\varphi_1$ and $C_2$ tracks the slower tone $\varphi_2$. The difficulty, as we will see in Section 4, is the overlap region $I$.

### 2.3 Stopping criteria

The sifting (2) is an *infinite* iteration in principle; a stopping rule is required. Two criteria are standard.

**Cauchy / standard-deviation (SD) criterion** [1]. Let $h^{(k)}$ denote the $k$-th sifting iterate (so $h^{(0)}=r^{(0)}$ and $h^{(k+1)}$ is the result of one application of (2)). The sifting stops when the normalized squared difference between two successive iterates,

$$
\mathrm{SD}_k \;=\; \frac{\displaystyle\sum_{t}\big(h^{(k)}(t)-h^{(k-1)}(t)\big)^2}{\displaystyle\sum_{t}\big(h^{(k-1)}(t)\big)^2},
\tag{4}
$$

falls below a prescribed threshold; Huang et al. [1] recommend $\mathrm{SD}\in[0.2,0.3]$, and note that this is a *very rigorous* limit, comparable to the change in a Fourier spectrum produced by shifting the data by only a few samples.

**$S$-number criterion** [1,9]. The sifting stops when, for $S$ consecutive iterations, (i) the number of zero crossings equals the number of extrema or differs by at most one, and (ii) this count is *stable*. Extensive testing suggests $S\in[3,8]$, with the lower end preferred. The optimal $S$ can also be read off from a *confidence-limit* study: decompose the same data with different $S$ and different stopping rules, and take the spread of the resulting IMFs as the uncertainty; the value of $S$ at which the spread stops decreasing is the "optimal" one [9].

Both criteria are, in the language of numerical analysis, **a posteriori** fixed-point/Cauchy tests for the nonlinear iteration (2). Neither has a rigorous a priori justification that ties the threshold to the error in the extracted IMF, and both make the decomposition *sensitive to local perturbations and to the amount of data* [2,10]: changing a few samples at the beginning of the record, or changing the stopping threshold, can change the entire set of IMFs. This sensitivity is one of the motivations for the ensemble methods.

### 2.4 Boundary (end) effects

Because the envelopes (2) are splines *interpolating* the observed extrema, they are least reliable near the endpoints, where there is no extremum to anchor the spline on one side. The result is an *end effect*: the extracted IMFs are distorted near $t=0$ and $t=T$, and the distortion propagates inward over a distance comparable to the longest local wavelength. A variety of ad hoc remedies exist (endpoint extension, mirroring, boundary conditions on the spline), but none is universally satisfactory, and the end effect is a persistent, unavoidable feature of the spline-envelope construction.

---

## 3. A functional-analytic reformulation

The previous section described EMD as a recipe. For an analyst, the useful object is the **operator** that the recipe implements, and the question is what can be said about it in a function space.

### 3.1 EMD as a nonlinear iterative operator

Work in $X = L^2[0,T]$ (or $X=\ell^2$ for a discrete record). Define the **envelope operators**

$$
E_+,\,E_- : X \to X,
$$

mapping a signal to its upper and lower envelope (the cubic splines through its local maxima and minima), and the **mean-envelope operator**

$$
M \;=\; \tfrac{1}{2}\big(E_+ + E_-\big).
$$

The single sifting step (2) is the operator

$$
T \;=\; I - M,
\tag{5}
$$

and one mode extraction is the repeated application $h^{(k+1)}=T\,h^{(k)}$ until the stopping rule. The full EMD is the composition of $K$ such extractions, each applied to the residual left by the previous one.

The operators $E_\pm$ — and hence $M$ and $T$ — are **nonlinear** (they depend on the data's extremal structure) and are **not defined on all of $X$** (they require the presence of local extrema; a monotone or constant function has none). They are also **not contractive in any standard norm** that has been established; the sifting is not a projection, and $T$ does not commute with itself in a way that would make the iterates a simple spectral sequence. This is precisely why EMD, unlike the Fourier or wavelet transform, has no general convergence theorem in the literature: the iteration (5) is a *data-dependent nonlinear* iteration, and the tools that guarantee convergence of linear iterations (spectral radius $<1$, contractivity) do not directly apply.

### 3.2 The linear idealization: iterative filtering

The rigorous theory that *does* exist attaches to a **linear idealization** of the sifting [4,5,6,11]. The observation, made by Flandrin, Rilling, Gonçalves, and others, is that the mean-envelope subtraction (2) *behaves like a band-pass filter* whose frequency response is determined by the local spacing of the extrema: it passes the "fast" oscillations (spacings near the mean inter-extremum distance) and removes the slow trend. If one replaces the nonlinear $M$ by a **linear** band-pass/low-pass filter $P$ with frequency response $H(\omega)\in[0,1]$ (where $H$ is the fraction of the "fast" band removed per step), the sifting becomes the **linear iterative filtering** iteration

$$
x^{(k+1)} \;=\; (I - P)\,x^{(k)} \;=\; \big(I-P\big)^{k}x.
\tag{6}
$$

This is the *Iterative Filtering (IF)* method, proposed around 2011 as a principled alternative to the spline EMD [11], and it is the object to which the convergence theorems apply.

### 3.3 Convergence of the linear idealization

The convergence of (6) is a clean result in $L^2$ (or $\ell^2$). Decompose $x$ into Fourier modes; on a single mode at frequency $\omega$, the iteration (6) multiplies by $(1-H(\omega))$ per step, so the $k$-th iterate of that mode is $(1-H(\omega))^{k}x_\omega$, which tends to $0$ **if and only if**

$$
\big|1 - H(\omega)\big| \;<\; 1
\qquad\Longleftrightarrow\qquad
0 \;<\; H(\omega) \;<\; 2.
\tag{7}
$$

Because $0\le H\le 1$ for a genuine band-pass, (7) holds automatically on the pass-band, and the convergence is **geometric** with rate $|1-H(\omega)|$:

$$
\big\|(I-P)^{k}x\big\|_{L^2}
\;\le\;
\Big(\sup_\omega |1-H(\omega)|\Big)^{k}\,\|x\|_{L^2}.
\tag{8}
$$

The **first extracted IMF** is the component of $x$ in the band where $H\neq 0$ (the frequencies that are progressively removed); the residual retains the band where $H=0$. In the notation of the recent two-frequency analysis [11], the theorem states that, given a filter $w$ with normalized response $\widehat w$ and an aperiodic signal, if $|1-\widehat w(\xi)|<1$ (or $\widehat w(\xi)=0$), then the iterative filtering converges and the first IMF is exactly the band-pass component of the signal. The condition (7) is the *a priori* guarantee: it is a condition on the *filter*, not on the data, and it is verifiable in advance.

This is the rigorous core of the whole story, and it is worth stating its scope honestly: **the convergence theorem is for the linear filter (6), not for the nonlinear spline sifting (5).** The spline EMD is a *data-dependent approximation* of (6) whose effective $H(\omega)$ changes with the signal and with the iteration. The theorem tells us that *if* the spline sifting can be bounded by, or approximated by, a linear filter satisfying (7), then it inherits convergence; but bounding the nonlinear operator by a linear one, uniformly over the data, is not established in general.

### 3.4 The dyadic (octave-band) character

A further and important result of Flandrin, Rilling, and Gonçalves [4] is that the EMD sifting behaves, on average, as a **dyadic** (octave-band) filter: each IMF occupies a frequency band that is roughly one octave wide, and successive modes occupy successively lower octaves. This was established by decomposing a wideband *fractional Gaussian noise* (fGn) model with spectrum

$$
S(f)\;\propto\;|f|^{\,1-2H},\qquad 0<H<1,
\tag{9}
$$

where $H$ is the Hurst exponent: the EMD of the fGn produces a cascade of octave-band modes, and the modes are "significant" (i.e., carry signal rather than just the $1/f$-like noise) precisely where the empirical, mode-by-mode spectrum deviates from the fGn prediction (9). The upshot is that EMD is a **log-frequency multiresolution analysis** — structurally close to a wavelet or to a constant-$Q$ (log-frequency) short-time Fourier analysis — but *nonlinear and data-adaptive*, with the octave boundaries determined by the data rather than fixed in advance. This is the sense in which EMD is "the natural basis" for nonlinear, nonstationary oscillations: it adapts the scale (wavelength) to the local content.

This dyadic property is not merely descriptive; it is *load-bearing* for CEEMDAN (Section 6.1), whose adaptive-noise construction relies on the $k$-th mode of white noise occupying the same octave band as the $k$-th mode of the signal.

### 3.5 What is, and is not, proven

To summarize the state of the theory for the true (spline) EMD:

- **Proven:** convergence of the *linear* iterative-filtering idealization (6) under the condition (7) [4,5,6,11]; the dyadic/octave-band character of the sifting on average [4]; a non-smooth convex-optimization reformulation of a related sifting with convergence guarantees [6].
- **Not proven (in general):** convergence of the fully nonlinear spline sifting (5); existence and uniqueness of the IMF set; a rigorous error bound tying the stopping threshold (4) to the error in the extracted mode; a separation theorem quantifying *when* EMD can resolve two close-by frequencies (the "one or two frequencies" question, addressed only recently and only for the linear IF method [11]).

The gap between "proven for the linear idealization" and "not proven for the true nonlinear operator" is the central theoretical limitation of EMD, and it is the gap that the ensemble methods are, in effect, a workaround for.

---

## 4. Pathologies: mode mixing, non-uniqueness, and the ad-hoc stopping rule

### 4.1 Mode mixing

**Mode mixing** [2,10] is the central pathology. It occurs in two related forms:

- *One mode, two frequencies:* a single IMF contains oscillations at two distinct instantaneous frequencies, because the sifting cannot decide which is "fastest" where the two overlap.
- *One frequency, two modes:* a single physical frequency is split across two or more IMFs.

In **running example (1)**, mode mixing is *guaranteed to appear* in the overlap region $I$, where $\varphi_1'(t)\approx\varphi_2'(t)$. There, the two tones are locally indistinguishable on the inter-extremum scale, so the sifting (2) cannot cleanly separate them: the first IMF $C_1$ will "wobble" between tracking $\varphi_1$ and tracking $\varphi_2$, and the envelope mean will be contaminated by the slower tone. The result is an instantaneous frequency $\omega_{C_1}(t)$ that jumps between the two values over $I$, and a mode $C_1$ that is not a genuine single oscillation — the very condition (1) of Section 2.1 fails locally. Outside $I$, where the tones are separated, the sifting behaves as intended.

Mode mixing is not a bug that can be patched by a better spline or a better stopping rule; it is a *consequence of the nonlinearity and data-adaptivity*. The sifting has no *a priori* notion of a frequency band to enforce; it only knows the local extrema, and where two tones overlap, the local extrema carry the mixture.

### 4.2 Non-uniqueness and sensitivity

Because the stopping rule (4) or the $S$-number is a threshold, and because the sifting is nonlinear, the decomposition is **not unique**: different thresholds, different spline endpoint treatments, or a few extra/missing samples produce different IMF sets [2,10]. There is no canonical "the" EMD of a signal; there is a *family* of decompositions parameterized by the stopping rule and the endpoint treatment. This non-uniqueness is what the confidence-limit study [9] measures (the spread over the family) and what the ensemble methods aim to *average out*.

---

## 5. Ensemble EMD (EEMD)

### 5.1 The noise-assisted ensemble

Wu and Huang's **Ensemble Empirical Mode Decomposition (EEMD)** [2] regularizes the nonlinear operator by *stochastic perturbation*. The algorithm is:

1. Choose an ensemble size $I$ and a noise amplitude $\sigma$ (typically a fixed fraction, e.g. $0.1$, of the standard deviation of $x$).
2. Draw $I$ independent white-noise realizations $w^i(t)$, $i=1,\dots,I$, each with standard deviation $\sigma$.
3. For each $i$, compute the **full** EMD of the noisy signal $x+w^i$, obtaining its complete set of IMFs $\{\mathrm{IMF}^{i}_k\}_{k=1}^{K_i}$.
4. The $k$-th mode of the *original* signal is the **ensemble mean**
$$
C_k \;=\; \frac{1}{I}\sum_{i=1}^{I}\mathrm{IMF}^{i}_k,
\qquad
R \;=\; x - \sum_{k} C_k.
\tag{10}
$$

The white noise serves two purposes. First, it *decorrelates* the mode mixing: adding independent noise to each copy perturbs the local extrema differently in each member, so the mixing "wobble" in $C_1$ is *independent* across members and cancels in the ensemble mean (10). Second, it makes the decomposition *robust*: the mean over many independent perturbations is far less sensitive to the stopping rule and to endpoint effects than any single decomposition.

In **running example (1)**, the effect of EEMD is that, in the overlap region $I$, the per-member $C_1^{i}$ still wobbles between $\varphi_1$ and $\varphi_2$, but the *mean* $C_1$ is a stable, well-defined oscillation (roughly the energy-weighted combination of the two tones over $I$), and the instantaneous frequency $\omega_{C_1}(t)$ is smooth rather than jumping. The mode mixing is not *eliminated* (the two tones are genuinely not separable in $I$ by any method that only sees the data), but it is *averaged out* into a consistent, interpretable mode.

A subtlety that EEMD does not resolve, and that CEEMDAN will, is that **different members may yield a different number of modes** $K_i$ [3]. The ensemble mean (10) is then ill-defined unless one pads the shorter decompositions with zeros, which injects a residual noise and breaks the exact reconstruction. This is the "incompleteness" of EEMD, quantified in Section 6.2.

### 5.2 The $\boldsymbol{1/\sqrt{I}}$ error reduction

The error analysis of EEMD is a **Monte-Carlo / central-limit** argument [2]. Write the $k$-th mode estimate (10) as

$$
C_k \;=\; \overline{C_k} \;+\; \frac{1}{I}\sum_{i=1}^{I}\big(\mathrm{IMF}^{i}_k - \overline{C_k}\big),
$$

where $\overline{C_k}$ is the (unknown) true mode. If the per-member deviations $\mathrm{IMF}^{i}_k-\overline{C_k}$ are approximately independent with variance $\sigma^2_{\mathrm{IMF}}$ (a reasonable assumption for independent white-noise perturbations), then the variance of the estimate is $\sigma^2_{\mathrm{IMF}}/I$ and the standard deviation of the error scales as

$$
\mathrm{std}\big(C_k - \overline{C_k}\big) \;\propto\; \frac{1}{\sqrt{I}}.
\tag{11}
$$

This is the standard Monte-Carlo scaling: the ensemble mean converges to the true mode at rate $1/\sqrt{I}$. Wu and Huang [2] establish precisely this: for a fixed noise amplitude, increasing the ensemble size $I$ reduces the standard deviation of the error in the ensemble-mean modes, and they give guidance that the noise amplitude $\sigma$ should be chosen relative to the signal and that $I$ should be large (hundreds to thousands) for a reliable decomposition. The scaling (11) is the *asymptotic* (law-of-large-numbers) result; a sharp, non-asymptotic bound for the *nonlinear* operator is not available, and the "independence" of the per-member deviations is an assumption, not a theorem (the sifting is nonlinear, so the per-member modes are not strictly independent of the noise in a way that a CLT directly covers). This is the honest limit of the EEMD error theory.

A concrete consequence, made explicit in [3], is the **residual-noise floor** of an *incomplete* EEMD: because the noise $w^i$ is not explicitly removed and the mode count varies, the ensemble-mean reconstruction carries a residual noise of standard deviation

$$
\varepsilon_r \;=\; \frac{\sigma}{\sqrt{I}},
\tag{11a}
$$

so that recovering a reconstruction error below a target $\delta$ would require $I \gtrsim (\sigma/\delta)^2$ realizations. For $\sigma=0.2$ and a machine-precision target $\delta\sim 10^{-15}$, this is $I\gtrsim 10^{29}$ — computationally absurd. This is the quantitative sense in which EEMD is "incomplete," and it is the motivation for CEEMDAN's exact reconstruction (14).

### 5.3 Dyadic filter-bank interpretation

EEMD has a clean interpretation as a **randomized dyadic filter bank** [2,4]. Each ensemble member is a dyadic (octave-band) decomposition — by Section 3.4 — but the octave boundaries are *shifted* by the independent noise, so that each member samples a slightly different dyadic partition. The ensemble mean (10) is then a *coherent sum* over an ensemble of dyadic partitions, which averages out the partition-dependent (i.e., mixing-dependent) part and retains the common (i.e., signal) part. This is the filter-bank analogue of (11): the noise randomizes the filter boundaries, and the ensemble average is a variance-reduced estimate of the true band-pass components.

### 5.4 Cost and parameter choice

The computational cost of EEMD is $I$ **full** EMD decompositions (each with $K_i$ modes and $S$ sifting iterations per mode), i.e. of order $I\cdot K_i\cdot S$ envelope operations. This is the main practical limitation: a reliable EEMD often needs $I$ in the hundreds to thousands, so the cost is $I$ times that of a single EMD. The two free parameters are $\sigma$ (noise amplitude) and $I$ (ensemble size): too small a $\sigma$ does not decorrelate the mixing, too large a $\sigma$ injects noise that the ensemble cannot fully average; too small an $I$ leaves residual variance (11). There is no rigorous, data-driven choice of $(\sigma,I)$; in practice they are set by a target signal-to-noise ratio and a target error tolerance.

---

## 6. CEEMDAN: Complete EEMD with Adaptive Noise

### 6.1 The adaptive-noise construction

Torres, Colominas, Schlotthauer, and Flandrin [3] introduced **CEEMDAN** (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise), which modifies EEMD in two ways: (i) a *single, consistent* residual is carried through all stages (rather than one residual per ensemble member), and (ii) the noise injected at each stage is not raw white noise but the **$k$-th EMD mode of the white-noise realization**, scaled by a coefficient $\varepsilon_k$. The algorithm, as given in [3], is:

**Input:** signal $x[n]$, ensemble size $I$, noise coefficients $\varepsilon_k$ (chosen to fix the SNR at each stage).

1. **First mode.** Decompose by EMD the $I$ realizations $x[n] + \varepsilon_0\, w^i[n]$, $i=1,\dots,I$, where $w^i$ is white noise with $\mathcal{N}(0,1)$; obtain their first modes and set
$$
\widetilde{\mathrm{IMF}}_1[n] \;=\; \frac{1}{I}\sum_{i=1}^{I} E_1\big(x[n]+\varepsilon_0\, w^i[n]\big),
\qquad
\widetilde{r}_1[n] \;=\; x[n] - \widetilde{\mathrm{IMF}}_1[n].
\tag{12}
$$
Here $E_1(\cdot)$ denotes the operator that extracts the *first* EMD mode.

2. **Second mode.** Decompose the $I$ realizations $\widetilde{r}_1[n] + \varepsilon_1\, E_1(w^i[n])$ until their first EMD mode, and set
$$
\widetilde{\mathrm{IMF}}_2[n] \;=\; \frac{1}{I}\sum_{i=1}^{I} E_1\big(\widetilde{r}_1[n]+\varepsilon_1\, E_1(w^i[n])\big).
\tag{13a}
$$

3. **$k$-th mode** ($k=2,\dots,K$). Define the $k$-th residual
$$
\widetilde{r}_k[n] \;=\; \widetilde{r}_{k-1}[n] - \widetilde{\mathrm{IMF}}_k[n],
\tag{13b}
$$
then decompose the $I$ realizations $\widetilde{r}_k[n] + \varepsilon_k\, E_k(w^i[n])$ until their first EMD mode, and set
$$
\widetilde{\mathrm{IMF}}_{k+1}[n] \;=\; \frac{1}{I}\sum_{i=1}^{I} E_1\big(\widetilde{r}_k[n]+\varepsilon_k\, E_k(w^i[n])\big).
\tag{13}
$$

4. **Stop** when the residual $\widetilde{r}_K$ has fewer than two extrema (no further IMF can be extracted).

**Output:** the modes $\widetilde{\mathrm{IMF}}_1,\dots,\widetilde{\mathrm{IMF}}_K$ and the final residual
$$
R[n] \;=\; x[n] - \sum_{k=1}^{K}\widetilde{\mathrm{IMF}}_k[n],
\qquad
x[n] \;=\; \sum_{k=1}^{K}\widetilde{\mathrm{IMF}}_k[n] + R[n].
\tag{14}
$$

The defining feature is the **adaptive-noise term** $\varepsilon_k\,E_k(w^i[n])$ in (13): the noise added at stage $k$ is *not* raw white noise of a fixed amplitude (as in EEMD), but the **$k$-th EMD mode of the white-noise realization** $w^i$, scaled by $\varepsilon_k$. This is the "adaptive" in CEEMDAN, and it rests on the dyadic/octave-band property of the EMD filter bank [4,8]: by construction, the $k$-th mode of the white noise occupies the *same octave band* as the $k$-th mode of the signal, so the noise is *frequency-matched* to the mode being extracted at each stage. The coefficients $\varepsilon_k$ allow one to select the SNR at each stage independently; in [3] a fixed SNR is used across all stages.

> **Note on terminology.** The algorithm above is the one given in the original ICASSP 2011 paper [3]. A closely related variant, later also called "CEEMDAN" in the literature and in most open-source toolboxes, replaces $E_k(w^i)$ by raw noise scaled to the standard deviation of the current first-mode ensemble (or of the residual). Both share the "complete" reconstruction property (14) and the adaptive-noise idea; they differ only in the precise form of the injected noise. The ICASSP 2011 formulation (13) is the one analyzed here because it is the primary source and the one in which the "adaptive" character is most sharply expressed.

### 6.2 Why "complete"

CEEMDAN is called **complete** because the decomposition **reconstructs the signal exactly** (14), in contrast to EEMD. In EEMD, each member $x+w^i$ is decomposed *independently* in full, and the noise $w^i$ is *discarded*: it is not explicitly removed from the residual, and — critically — different realizations may produce a *different number of modes* $K_i$, so the ensemble mean (10) must be padded with zeros for the missing modes, introducing a residual noise of standard deviation $\varepsilon_r = \sigma/\sqrt{I}$ [3]. The reconstruction in EEMD is therefore *inexact*: the sum of the ensemble-mean modes does not return $x$ exactly, and the residual-noise floor (11a) sets a hard limit on the achievable reconstruction error for a given $I$.

In CEEMDAN, by contrast, a **single, consistent residual** $\widetilde{r}_k$ is carried through all stages, and the telescoping sum (14) holds *exactly* by construction: each mode is subtracted from the same residual, so no noise is left behind and no padding is needed. The noise is *retained* in the residual and *re-injected* at each level in the frequency-matched form $\varepsilon_k\,E_k(w^i)$, so the noise budget is consistent across all levels. This internal consistency is what "complete" denotes, and it is what makes CEEMDAN's modes more stable and less mode-mixed than EEMD's at the same $I$.

The reconstruction error in [3] is verified numerically to be at machine precision (max amplitude $<2\times10^{-15}$, standard deviation $2\times10^{-14}$ for an ECG signal), whereas matching that precision with EEMD would require $I\gtrsim 10^{29}$ realizations by (11a) — an impossibility.

In **running example (1)**, CEEMDAN's advantage over EEMD appears at the *deeper* modes. In the overlap region $I$, the first mode $\widetilde{\mathrm{IMF}}_1$ is the same stable, averaged oscillation in both EEMD and CEEMDAN. But when extracting $\widetilde{\mathrm{IMF}}_2$ (the slower tone), EEMD decomposes the *original* noisy signal in full, so the noise in $\widetilde{\mathrm{IMF}}_2$ is still at the original amplitude $\sigma$ — mismatched to the smaller energy of the second residual — whereas CEEMDAN injects $\varepsilon_1\,E_1(w^i)$, whose octave band is matched to the *second* mode by the dyadic property. The result is that CEEMDAN's $\widetilde{\mathrm{IMF}}_2$ is a cleaner estimate of the slower tone $\varphi_2$, with less residual-noise contamination, and the mode mixing in the overlap region is more effectively suppressed.

### 6.3 Comparison with EEMD

The distinction between EEMD and CEEMDAN can be summarized as follows:

| Aspect | EEMD [2] | CEEMDAN [3] |
|---|---|---|
| Noise per member | fresh white noise, fixed $\sigma$ | $k$-th EMD mode of the noise, $E_k(w^i)$, scaled by $\varepsilon_k$ |
| Decomposition per member | **full** EMD (all modes), independently | **first IMF only**, per stage, on a shared residual |
| Residual | one per member (inconsistent) | single, shared (consistent) |
| Noise accounting | discarded (inexact reconstruction, floor (11a)) | retained and re-injected (exact reconstruction (14)) |
| Mode-mixing control | by ensemble averaging | by ensemble averaging **+** frequency-matched adaptive noise |
| Modes at same $I$ | more noise-contaminated at deep levels | cleaner, less mode-mixed |
| Cost | $I\cdot K_i\cdot S$ envelope ops | $K\cdot I\cdot S$ envelope ops (comparable order) |
| Effective $I$ needed | larger | smaller for equivalent quality |

The computational orders are comparable (both are $I$ times a per-level cost), but CEEMDAN typically achieves a given quality with a *smaller* $I$, because the adaptive noise matching makes each ensemble member a better estimate — the variance reduction is more efficient than in EEMD. In [3], for a 512-sample Dirac signal, CEEMDAN required about half the sifting iterations of EEMD at $I=500$, and for a 10-second ECG signal only $30.8\%$ of the EEMD iterations. The practical consequence is that CEEMDAN is often the preferred method when the signal has well-separated scales and one wants a clean, internally-consistent decomposition with less mode mixing, at a cost comparable to (and often lower than) EEMD.

---

## 7. A unified numerical-analysis viewpoint

The three methods are best understood as a single object — a nonlinear decomposition operator — refined in three stages, each with a precise numerical-analysis interpretation.

### 7.1 (Regularized) nonlinear operators on $L^2/\ell^2$

- **EMD** is the raw nonlinear operator $T=I-M$ (5), iterated. It is a data-dependent, non-projection, non-contractive operator on a *subset* of $X$ (the functions with enough extrema). It has no general convergence theorem.
- **EEMD** is $T$ **regularized by Monte-Carlo averaging**: one evaluates the (nonlinear) mode-extraction map at $I$ independent noise-perturbed inputs and averages. The regularized operator is the *expected value* of the perturbed output, and its error is controlled by the variance of the perturbed outputs (11).
- **CEEMDAN** is the same Monte-Carlo regularizer, but with a **feedback** on the noise: the perturbation at stage $k$ is the $k$-th EMD mode of the noise, $E_k(w^i)$, frequency-matched to the $k$-th mode by the dyadic property, and scaled to the residual energy via $\varepsilon_k$ (13). It is a *closed-loop* (feedback) version of the open-loop EEMD regularizer.

In the language of numerical analysis, EMD is an *unregularized* nonlinear fixed-point iteration; EEMD is a *Monte-Carlo* (stochastic) approximation to its output; CEEMDAN is a *stochastic* approximation with *adaptive step-size* (the noise amplitude is the step-size, adapted to the residual and to the octave band). This is an apt analogy: the noise amplitude plays the role of a step-size that is scaled to the local residual, exactly as in an adaptive iterative method.

### 7.2 Monte-Carlo variance reduction for a nonlinear operator

The error of EEMD and CEEMDAN is, to first order, a **variance-reduction** problem. The per-member modes $\mathrm{IMF}^{i}_k$ are random variables (random in the noise), and the ensemble mean is their empirical mean. The standard Monte-Carlo result — variance $\propto 1/I$, standard deviation $\propto 1/\sqrt{I}$ — applies *conditionally* on the (approximate) independence of the per-member deviations, which is an assumption for the nonlinear operator but a reasonable one for independent white-noise perturbations. The $1/\sqrt{I}$ scaling (11) is therefore the correct *order* of the convergence, even if a sharp non-asymptotic bound for the nonlinear operator is not available. CEEMDAN's advantage is that its per-member deviations have *smaller* variance (the adaptive noise matching makes each member a better estimate), so the constant in (11) is smaller and a smaller $I$ suffices.

### 7.3 The stopping criterion as a Cauchy/fixed-point test

The SD criterion (4) and the $S$-number are, in numerical-analysis terms, **a posteriori** Cauchy/fixed-point tests for the nonlinear iteration (5). They measure the change between successive iterates and stop when it is small — the standard *a posteriori* criterion for a fixed-point iteration. What they lack is an *a priori* bound tying the threshold to the *error in the extracted mode* (as opposed to the *change between iterates*), and this is the sense in which the stopping rule is ad hoc. The ensemble methods do not remove this problem — each member is still sifted with the same ad-hoc rule — but they *distribute* the sensitivity: because the stopping-rule error is (approximately) independent across members, it is averaged out in the ensemble mean (10)/(13), exactly as the mode-mixing error is.

### 7.4 Relation to wavelets and other multiresolution methods

The three methods sit in the broader family of multiresolution analyses, and the contrast with the linear methods is instructive:

- **Wavelets / STFT:** linear, fixed, shift-invariant basis; exact convergence and stability (the basis is a frame/Riesz basis); the scale is *fixed in advance* (dyadic for wavelets).
- **EMD:** nonlinear, data-adaptive, no fixed basis; the scale is *adapted to the data* (dyadic on average, Section 3.4); no general convergence theorem; the mode-mixing and non-uniqueness are intrinsic.
- **EEMD / CEEMDAN:** the nonlinear EMD operator *regularized* by stochastic perturbation; the convergence is the *Monte-Carlo* convergence of the regularized (averaged) operator, at rate $1/\sqrt{I}$; the price is a stochastic (not deterministic) error and an $O(I)$ cost.

The deep point for the analyst is that EMD/EEMD/CEEMDAN are the *nonlinear, data-adaptive* counterpart of the wavelet/STFT multiresolution analysis, and that the ensemble methods are the *stochastic regularization* that restores — at a Monte-Carlo rate and an $O(I)$ cost — much of the robustness and uniqueness that the linear methods enjoy by construction.

### 7.5 Summary of what is rigorous

| Claim | Status |
|---|---|
| Convergence of the linear iterative-filtering idealization (6) under (7) | **Proven** [4,5,6,11] |
| Dyadic/octave-band character of the sifting (on average) | **Proven** (fGn analysis) [4] |
| EEMD/CEEMDAN error $\propto 1/\sqrt{I}$ | **Proven in order** (Monte-Carlo/CLT, conditional on approximate independence) [2] |
| Exact (machine-precision) reconstruction of CEEMDAN | **Proven by construction** (telescoping (14)); verified numerically [3] |
| Convergence of the fully nonlinear spline sifting (5) | **Not proven** in general |
| Existence/uniqueness of the IMF set | **Not proven** in general |
| A priori bound tying the stopping threshold to the mode error | **Not proven** (ad-hoc, a posteriori rule) |
| Separation of two close-by frequencies by the true EMD | **Not proven** (only for the linear IF method, recently) [11] |

---

## 8. Current directions and open problems

The theory of EMD and its ensemble refinements is, as of the recent two-frequency analysis [11] and the convex-optimization reformulations [6], at the stage where the *linear idealization* is well understood but the *true nonlinear operator* is not. The principal open problems are:

1. **General convergence of the nonlinear sifting.** Establishing, for the spline operator $T=I-M$ (5), an *a priori* convergence and stability result that does not rely on bounding $T$ by a linear filter. The convex-optimization reformulation of a related sifting [6] is a promising route, because it recasts the sifting as a (non-smooth) convex minimization, for which standard convergence theorems apply; extending this to the exact spline sifting is an open problem.

2. **The two-frequency separation limit.** Quantifying, for the *nonlinear* EMD (not just the linear IF), the condition under which two close-by frequencies are resolvable — the "one or two frequencies" question [11]. For the linear method this is governed by the filter response (7); for the nonlinear method the condition must involve the data-dependent envelope structure, and no such condition is known.

3. **A rigorous stopping criterion.** Replacing the ad-hoc SD/$S$-number with an *a priori* rule whose threshold can be tied to the error in the extracted mode, and whose effect is *controllable* under the ensemble averaging of EEMD/CEEMDAN.

4. **Non-asymptotic error bounds for the ensemble methods.** The $1/\sqrt{I}$ scaling (11) is an asymptotic (CLT) result conditional on approximate independence; a *non-asymptotic*, distribution-free bound for the nonlinear operator — e.g. via concentration inequalities that do not require independence — is an open problem that would make the choice of $I$ principled rather than heuristic.

5. **Optimal noise scheduling in CEEMDAN.** The choice of the adaptive-noise coefficients $\varepsilon_k$ (13) — the SNR at each stage — is a natural, heuristic choice (fix the SNR, as in [3]); whether it is *optimal* in a variance-minimization sense, and whether a different schedule (e.g. one derived from a minimax bound, or one that matches the octave-band energy of the residual) can reduce the required $I$, is open. The original authors flag this explicitly as future work [3].

---

## 9. References

1. N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q. Zheng, N. C. Yen, C. C. Tung, H. H. Liu, "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis," *Proc. R. Soc. Lond. A* **454**(1971) (1998), 903–995.
2. Z. Wu, N. E. Huang, "Ensemble Empirical Mode Decomposition: a noise-assisted data analysis method," *Adv. Adapt. Data Anal.* **1**(1) (2009), 1–41.
3. M. E. Torres, M. A. Colominas, G. Schlotthauer, P. Flandrin, "A complete ensemble empirical mode decomposition with adaptive noise," in *Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, pp. 4144–4147, 2011.
4. P. Flandrin, G. Rilling, P. Gonçalves, "Empirical mode decomposition as a filter bank," *IEEE Signal Process. Lett.* **11**(2) (2004), 112–114.
5. G. Rilling, P. Flandrin, P. Gonçalves, "On empirical mode decomposition and its algorithms," in *IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing (NSIP)*, 2003.
6. N. Pustelnik, P. Borgnat, P. Flandrin, "Empirical mode decomposition revisited by multicomponent non-smooth convex optimization," *Signal Process.* **102** (2014), 313–331.
7. C. C. Huang, L. Yang, Y. Wang, "Convergence of a convolution-filtering-based algorithm for empirical mode decomposition," *Adv. Adapt. Data Anal.* **1**(4) (2009), 561–571.
8. P. Flandrin, P. Gonçalves, G. Rilling, "EMD Equivalent Filter Banks, from Interpretation to Applications," in *Hilbert–Huang Transform and Its Applications*, N. E. Huang and S. Shen (eds.), World Scientific, 2005, pp. 57–74.
9. N. E. Huang, Z. Wu, H. Long, O. P. Spencer, "A study of truncation effects in the Hilbert–Huang method," *Phil. Trans. R. Soc. Lond. A* **361** (2003), 1307–1324 ($S$-number and confidence-limit criteria).
10. N. E. Huang, Z. Wu, *Hilbert–Huang Transform and Its Applications*, Wiley-Interscience, 2014.
11. A. Cicone, S. Serra-Capizzano, H. Zhou, "One or two frequencies? The Iterative Filtering answers," *arXiv:2111.11741* (2021); also *Appl. Math. Comput.* **462** (2024), 128322.
12. P. Flandrin, P. Gonçalves, G. Rilling, "Time–frequency analysis of nonstationary signals in the EMD framework," *IEEE Trans. Signal Process.* **55**(4) (2007), 1336–1344.

---

*A note on scope and honesty.* Throughout this essay the distinction between what is **proven** (the convergence of the linear iterative-filtering idealization; the dyadic character; the $1/\sqrt{I}$ Monte-Carlo scaling; the exact reconstruction of CEEMDAN by construction) and what is **heuristic or open** (the convergence of the fully nonlinear spline sifting; the existence/uniqueness of the IMF set; the optimality of the stopping rule and the CEEMDAN noise schedule; the two-frequency separation limit for the true EMD) is maintained deliberately. The ensemble methods are, in effect, a *practical* workaround for the lack of a general nonlinear convergence theory: they restore robustness and (approximate) uniqueness at a Monte-Carlo rate and an $O(I)$ cost, but they do not remove the fundamental nonlinearity that is the source of the theoretical gap.
