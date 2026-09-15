# The Hilbert Transform and Bedrosian's Theorem: A Spectral-Commutator Account

## 1. Scope and thesis

The Hilbert transform is one of the oldest objects in analysis — a singular integral that has, over a century, acquired at least four distinct identities: as the conjugate function of a harmonic extension, as the boundary value of a Hardy-space projection, as a $90^\circ$ phase shifter in signal processing, and as the kernel of the Kramers–Kronig dispersion relations in physics. Each community meets the operator through a different door.

Bedrosian's theorem is the hinge on which these doors turn. In its engineering form it says that the Hilbert transform of a product of a low-pass and a high-pass signal, whose spectra do not overlap, is the product of the low-pass signal and the Hilbert transform of the high-pass signal. In its analytical form it is a statement about when the Riesz projection commutes with a multiplication operator. This essay argues that **Bedrosian's theorem is not an isolated "product rule" for the Hilbert transform but the precise spectral-commutator condition under which the Riesz projection $P=\frac12(I+iH)$ commutes with multiplication by a band-limited factor**, and that the familiar "non-overlapping spectra" hypothesis is exactly the sharp necessary and sufficient condition for that commutation. Once this is made explicit, the theorem stops being a clever trick for demodulating amplitude-modulated signals and becomes an instance of a general principle about Fourier multipliers and the geometry of their symbols, with clean consequences for numerical computation, for the Kramers–Kronig relations, and for multidimensional extensions.

The intended reader works in functional and numerical analysis, mathematical physics, signal analysis, or system theory, so the development is stated at the level of $L^2$ and Hardy spaces, with proofs of the main identity and of the sharpness of the hypothesis, and with explicit worked applications.

---

## 2. The Hilbert transform: definitions and functional-analytic setup

We work on $\mathbb R$ with the unitary Fourier transform

$$\hat f(\xi)=\mathcal F[f](\xi)=\int_{\mathbb R} f(t)\,e^{-i\xi t}\,dt,\qquad f(t)=\frac1{2\pi}\int_{\mathbb R}\hat f(\xi)\,e^{i\xi t}\,d\xi.$$

With this convention the Hilbert transform is the convolution with the singular kernel $1/(\pi t)$,

$$Hf(t)=\mathcal H[f](t)=\frac1\pi\,\mathrm{p.v.}\int_{\mathbb R}\frac{f(t-s)}{s}\,ds,$$

and Plancherel reduces it to the Fourier multiplier

$$\widehat{Hf}(\xi)=-i\,\mathrm{sgn}(\xi)\,\hat f(\xi),\qquad \mathrm{sgn}(0):=0.$$

Three facts about $H$ will be used throughout.

**(i) $L^2$-isometry and $L^p$-boundedness.** On $L^2(\mathbb R)$ the operator is an isometry, $\|Hf\|_2=\|f\|_2$, since $|-i\,\mathrm{sgn}|=1$ a.e. On $L^p(\mathbb R)$ it is bounded for $1<p<\infty$ with norm $\|H\|_{p\to p}=\lvert\csc(\pi/p)\rvert$ (the M. Riesz theorem, $1928$), and it is unbounded on $L^1$ and $L^\infty$ but of weak type $(1,1)$ and $(\infty,\infty)$, so it is of restricted type at the endpoints by the Marcinkiewicz interpolation theorem. See Stein, *Singular Integrals and Differentiability Properties of Functions* (1970), Ch. II.

**(ii) $H$ is a complex structure.** $H^2=-I$ on $L^2(\mathbb R)$ (off the null set $\{0\}$, which does not affect $L^2$ functions). Thus $H$ is a skew-adjoint, self-inverse-up-to-sign operator: it rotates the positive-frequency half of a signal by $+\pi/2$ and the negative-frequency half by $-\pi/2$. This is the "quadrature" or "conjugate" role that makes the analytic signal possible.

**(iii) $H$ generates the analytic signal via the Riesz projection.** Define the Riesz (or Cauchy) projection

$$P f=\frac12(f+iHf),\qquad \widehat{Pf}(\xi)=\chi_{(0,\infty)}(\xi)\,\hat f(\xi).$$

$P$ is the orthogonal projection of $L^2(\mathbb R)$ onto the Hardy space $H^2(\mathbb R)$, the subspace of functions whose Fourier transforms are supported in $(0,\infty)$. For a *real* signal $u\in L^2(\mathbb R)$ (so $\hat u(-\xi)=\overline{\hat u(\xi)}$), the **analytic signal** is

$$u_a:=u+iHu=(I+iH)u,\qquad \widehat{u_a}(\xi)=2\,\chi_{(0,\infty)}(\xi)\,\hat u(\xi)+\chi_{\{0\}}(\xi)\hat u(\xi),$$

i.e. $u_a$ is the $L^2$ function whose spectrum is the positive-frequency half of $\hat u$ (doubled) with the DC term left intact. The real part of $u_a$ is $u$ and the imaginary part is $Hu$; the modulus $|u_a|$ is the **envelope** and $\arg(u_a)$ is the **instantaneous phase**, whose derivative is the **instantaneous frequency**. The entire Hilbert–Huang transform, and much of time–frequency analysis, rests on this construction.

The Bedrosian theorem governs precisely when the analytic-signal operation commutes with multiplication — the operation that produces amplitude modulation. We turn to it next.

---

## 3. Statement of Bedrosian's theorem

**Theorem (Bedrosian, 1963; sharp form, Lin–Zhang, 2014/2016).** Let $f,g\in L^2(\mathbb R)$. The identity

$$\boxed{\;\mathcal H[fg]=f\,\mathcal H[g]\;}$$

holds for every $g\in L^2(\mathbb R)$ with a given support condition if and only if there exist $0\le a,b<\infty$ with

$$\operatorname{supp}\hat f\subset[-a,b]\quad\text{and}\quad \operatorname{supp}\hat g\subset\mathbb R\setminus(-b,a).$$

In the engineering language, $f$ is a **low-pass** signal whose spectrum lies in the band $[-a,b]$ and $g$ is a **high-pass** signal whose spectrum avoids the interval $(-b,a)$; the two spectra are separated by a (possibly zero-width) gap. In the common case $a=b=\omega_0>0$, the condition reads $\operatorname{supp}\hat f\subset[-\omega_0,\omega_0]$ and $\operatorname{supp}\hat g\subset\mathbb R\setminus(-\omega_0,\omega_0)$: the carrier sits at $|\xi|\ge\omega_0$ while the envelope is confined to $|\xi|\le\omega_0$.

The theorem is usually quoted in the weaker, sufficient direction (non-overlap $\Rightarrow$ identity). The contribution of the modern harmonic-analysis treatment is the **necessity** direction: if the support condition fails, one can find $f,g\in L^2$ for which the identity fails. We prove both directions below; the necessity is what elevates the result from a "rule of thumb" to a characterization.

---

## 4. Proof via the Riesz projection

Write $iH=2P-I$ (equivalently $H=-i(2P-I)$). The identity $H(fg)=fH(g)$ is equivalent to

$$(2P-I)(fg)=f(2P-I)g\;\;\Longleftrightarrow\;\;P(fg)=fPg,$$

because the $-I(fg)=-fg=-fIg$ terms cancel. Thus **Bedrosian's identity is exactly the commutation of the Riesz projection with multiplication by $f$**:

$$P M_f=M_f P.$$

This reframing is the conceptual core of the essay: the theorem is a commutator statement, not a "product rule."

Now pass to the frequency domain. Since $M_f$ is multiplication by $f$ in time, it is convolution by $\hat f$ in frequency, while $P$ is the multiplier $\chi_{(0,\infty)}$. The condition $P(fg)=fPg$ becomes

$$\chi_{(0,\infty)}\bigl(\hat f * \hat g\bigr) = \hat f * \bigl(\chi_{(0,\infty)}\hat g\bigr). \tag{$\star$}$$

Let $A=\operatorname{supp}\hat f$, and split the carrier spectrum into its positive and negative parts $B^+=\operatorname{supp}\hat g\cap(0,\infty)$ and $B^-=\operatorname{supp}\hat g\cap(-\infty,0]$. The left side of $(\star)$ is supported in $(A+B^+\cup A+B^-)\cap(0,\infty)$, whereas the right side is supported in $A+B^+$. Hence $(\star)$ holds if and only if

$$A+B^+\subset[0,\infty) \qquad\text{and}\qquad A+B^-\subset(-\infty,0], \tag{$\star\star$}$$

for in that case both sides reduce to $\hat f*(\chi_{(0,\infty)}\hat g)$ on $(0,\infty)$ and to $0$ elsewhere. The closed endpoints in $(\star\star)$ are immaterial: the point $\xi=0$ carries no $L^2$ mass, so the identity is unaffected by whether the sumset reaches $0$. Condition $(\star\star)$ says precisely that no translate of the envelope spectrum by a *positive* carrier frequency lands in the open negative half-line, and no translate by a *negative* carrier frequency lands in the open positive half-line: the sumset of the two spectra must not straddle $0$ in either direction.

**Sufficiency.** Suppose $\operatorname{supp}\hat f\subset[-a,b]$ and $\operatorname{supp}\hat g\subset\mathbb R\setminus(-b,a)$ for some $0\le a,b<\infty$. Then $B^+\subset[a,\infty)$ and $B^-\subset(-\infty,-b]$, so

$$A+B^+\subset[-a,b]+[a,\infty)=[0,\infty),\qquad A+B^-\subset[-a,b]+(-\infty,-b]=(-\infty,0].$$

Both inclusions in $(\star\star)$ hold, so $(\star)$, and therefore $H(fg)=fH(g)$, hold. $\square$

**Sharpness (necessity).** The separation is not a convenient over-condition. If the identity $H(fg)=fH(g)$ is to hold for a fixed $f$ and all $g$ whose spectrum lies in a set $B$, then $(\star\star)$ forces $B$ to avoid the open interval $(-b,a)$, where $a$ is set by the extent of $\operatorname{supp}\hat f$; and the full characterization of the admissible support sets for a general bounded Fourier multiplier — hence for $H$ — is exactly the interval form $A\subset[-a,b]$, $B\subset\mathbb R\setminus(-b,a)$ (Lin–Zhang, 2014/2016). The counterexample below shows the identity fails as soon as the separation is violated, even slightly.

**Remark (dual form).** By the same argument with the roles of $f$ and $g$ reversed (using $iH=2P-I$ and the conjugate projection $\overline{P}=\tfrac12(I-iH)$, which projects onto negative frequencies), one also obtains the companion identity

$$\mathcal H[fg]=g\,\mathcal H[f]$$

whenever $\operatorname{supp}\hat g\subset[-a,b]$ and $\operatorname{supp}\hat f\subset\mathbb R\setminus(-b,a)$. The two forms are the "low-pass on the left" and "high-pass on the left" variants, and either may be used depending on which factor is the slowly varying envelope.

---

## 5. Sharpness: a counterexample when spectra overlap

The necessity direction is not vacuous. Take $f=g=\operatorname{sinc}$ with $\operatorname{sinc}(t)=\frac{\sin t}{\pi t}$, so $\hat f=\hat g=\chi_{[-1,1]}$. Both spectra sit in $[-1,1]$, so the Bedrosian condition fails (there is no band separation; indeed $a=b=1$ would require $\operatorname{supp}\hat g\subset\mathbb R\setminus(-1,1)$, which is false). Here $A=[-1,1]$, $B^+=(0,1]$, $B^-= [-1,0)$, and $A+B^+=(-1,2]$ is **not** a subset of $[0,\infty)$ — the sumset straddles $0$ — so the condition $(\star\star)$ is violated and the identity must fail.

Compute in the frequency domain. The product $fg$ has the triangular spectrum

$$\widehat{fg}(\xi)=(\hat f*\hat g)(\xi)=\chi_{[-1,1]}*\chi_{[-1,1]}(\xi)=(2-|\xi|)\,\chi_{[-2,2]}(\xi),$$

hence

$$\widehat{H(fg)}(\xi)=-i\,\mathrm{sgn}(\xi)\,(2-|\xi|)\,\chi_{[-2,2]}(\xi).$$

On the other hand, $\widehat{Hg}(\xi)=-i\,\mathrm{sgn}(\xi)\,\chi_{[-1,1]}(\xi)$, so

$$\widehat{fH(g)}(\xi)=(\hat f*\widehat{Hg})(\xi)=-i\int_{[\xi-1,\xi+1]\cap[-1,1]}\mathrm{sgn}(s)\,ds,$$

the integral being the signed length of the overlap interval. Evaluate at $\xi=\tfrac12$: the overlap is $[\tfrac12-1,\tfrac12+1]\cap[-1,1]=[-\tfrac12,1]$, so

$$\widehat{fH(g)}\Bigl(\tfrac12\Bigr)=-i\Bigl(\int_{-1/2}^{0}(-1)\,ds+\int_{0}^{1}(1)\,ds\Bigr)=-i\Bigl(-\tfrac12+1\Bigr)=-\tfrac12\,i,$$

whereas

$$\widehat{H(fg)}\Bigl(\tfrac12\Bigr)=-i\cdot(+1)\cdot\Bigl(2-\tfrac12\Bigr)=-\tfrac32\,i.$$

The two differ ($-\tfrac32 i\ne-\tfrac12 i$), so $H(fg)\ne fH(g)$. By the same computation the discrepancy is confined to the open interval $(-1,1)$: on $[-2,-1]\cup[1,2]$ the two sides agree (both vanish at $\xi=\pm2$), while on $(-1,1)$ they differ. The failure is the mechanism the proof predicted — the translate of the envelope spectrum by a positive carrier frequency crosses $0$ — and it is not a measure-zero artifact: the two sides differ on a set of full measure in $(-1,1)$, so the identity fails in $L^2$.

---

## 6. Applications

### 6.1 Amplitude modulation and demodulation

Let $u_m$ be a message signal with $\operatorname{supp}\hat u_m\subset[-\omega_0,\omega_0]$ and let the transmitted signal be the carrier-modulated waveform

$$u(t)=u_m(t)\cos(\omega_c t),\qquad \omega_c>\omega_0.$$

The Bedrosian condition holds with $a=\omega_0$, $b=\omega_0$ (envelope in $[-\omega_0,\omega_0]$, carrier at $|\xi|=\omega_c>\omega_0$). Since $\mathcal F[\cos\omega_c t]=\pi(\delta_{\omega_c}+\delta_{-\omega_c})$,

$$\mathcal H[\cos\omega_c t]=\operatorname{sgn}(\omega_c)\sin\omega_c t,$$

so for $\omega_c>0$,

$$\mathcal H[u]=\mathcal H[u_m\cos\omega_c t]=u_m(t)\,\mathcal H[\cos\omega_c t]=u_m(t)\sin\omega_c t.$$

Therefore the analytic signal factors completely:

$$u_a=u+iHu=u_m\cos\omega_c t+i\,u_m\sin\omega_c t=u_m(t)\,e^{i\omega_c t}.$$

The envelope is $|u_a|=|u_m|$ and the instantaneous phase is $+\omega_c t$ (plus the argument of $u_m$ if $u_m$ is complex-valued), so the instantaneous frequency is $\omega_c$. This is the mathematical content of envelope detection: **the modulus of the analytic signal recovers the message envelope exactly, with no cross-term, precisely because the carrier and envelope spectra are separated.**

For standard AM, $u(t)=(1+\mu u_m(t))\cos\omega_c t$, the same computation gives $u_a=(1+\mu u_m)e^{i\omega_c t}$, envelope $|1+\mu u_m|$ and instantaneous frequency $\omega_c$. Over-modulation ($\mu>1$) drives $1+\mu u_m$ through zero, where the envelope vanishes and the instantaneous phase jumps by $\pi$ — the familiar over-modulation distortion, now seen as a phase singularity rather than an empirical artifact.

**The separation is essential.** If $\omega_c\le\omega_0$, the spectra overlap and the factorization $u_a=u_me^{i\omega_c t}$ fails: the Hilbert transform of the product acquires cross-terms that mix the carrier with the message, the envelope is no longer $|u_m|$, and the instantaneous frequency is contaminated. This is the precise sense in which a "low carrier" cannot be demodulated by the analytic-signal method.

### 6.2 Instantaneous frequency and the Hilbert–Huang transform

The Hilbert–Huang transform (Huang, Shen, Long, et al., 1998) decomposes a signal into intrinsic mode functions (IMFs) and reads the instantaneous frequency of each IMF as the derivative of the argument of its analytic signal. The validity of that readout for a single IMF rests on the same Bedrosian mechanism: an IMF is, by construction, a nearly monochromatic oscillation whose envelope is slowly varying relative to its carrier, so the support-separation hypothesis holds to good approximation and the analytic signal factors as (envelope)$\times$ (pure phase). Bedrosian's theorem is thus the local justification for the instantaneous-frequency concept on which the whole method is built; the decomposition is an algorithm for producing factors that satisfy the hypothesis, and the theorem is the reason the readout is then legitimate.

### 6.3 Kramers–Kronig and causality

For a causal, stable, linear time-invariant system with impulse response $g(t)=\chi_{[0,\infty)}g(t)$ and transfer function $G(\omega)=\mathcal F[g](\omega)=G_r(\omega)+iG_i(\omega)$, causality (together with the convention $\hat f(\xi)=\int f(t)e^{-i\xi t}dt$) implies that $G$ is the boundary value, from the **lower** half-plane, of a function holomorphic there. The Sokhotski–Plemelj theorem then gives the **Kramers–Kronig relations**:

$$G_r(\omega)=-\frac1{\pi}\,\mathrm{p.v.}\int_{\mathbb R}\frac{G_i(\omega')}{\omega'-\omega}\,d\omega',\qquad G_i(\omega)=\frac1{\pi}\,\mathrm{p.v.}\int_{\mathbb R}\frac{G_r(\omega')}{\omega'-\omega}\,d\omega',$$

i.e. each of the real and imaginary parts is, up to the sign dictated by the convention, the Hilbert transform of the other (the sign of each side flips if the opposite Fourier convention is used). The logical structure is identical to the analytic-signal construction: a *one-sided* (causal) object has a real part and an imaginary part that are mutually determined by the Hilbert transform. Bedrosian's theorem is the multiplicative counterpart of the same duality — where Kramers–Kronig relates the two parts of a single causal spectrum, Bedrosian relates the Hilbert transform of a *product* to the product of the Hilbert transform of one factor, under the condition that one factor's spectrum is "one-sided" relative to the other's. Both are expressions of the fact that the Hilbert transform is the boundary-value conjugation of a half-plane analyticity condition.

### 6.4 Numerical computation

The discrete Hilbert transform is computed by the FFT: for a sampled signal $x[n]$ with $N$-point DFT $X[k]$, the discrete Hilbert transform is $\mathcal F^{-1}[-i\,\mathrm{sgn}(k)\,X[k]]$ (with the DC and, for even $N$, the Nyquist bins set to zero). Bedrosian's theorem is what justifies the standard engineering shortcut in which one does **not** compute the full Hilbert transform of the modulated signal $u_m\cos\omega_c t$: because the carrier and envelope spectra are separated, the transform of the product is the product of the envelope and the transform of the carrier, so the analytic signal — and hence the envelope and instantaneous frequency — can be formed from the envelope alone.

In practice the separation is only approximate (finite records, windowing, and sampling all cause spectral leakage), and the accuracy of the analytic-signal readout degrades as the gap closes. A useful numerical diagnostic is therefore the **spectral gap**: the width of the frequency band separating $\operatorname{supp}\hat u_m$ and the carrier support. The larger the gap, the more accurately the Bedrosian factorization holds and the cleaner the instantaneous-frequency estimate. This makes the theorem directly relevant to the design of the window, the record length, and the carrier choice in any FFT-based Hilbert-transform pipeline. The discrete-time theory (the discrete analytic signal and its Bedrosian product theorem) was developed by Brown and others; see the discrete-time references below.

---

## 7. Extensions

**Fractional Hilbert transforms.** The fractional Hilbert transform $H^\alpha$ with multiplier $-i\,\mathrm{sgn}(\xi)^\alpha$ interpolates between the identity ($\alpha=0$) and the ordinary Hilbert transform ($\alpha=1$). Fractional analytic signals $f+iH^\alpha f$ inherit Bedrosian-type product properties, and their time–frequency localization is unchanged; see the fractional-Hilbert-transform literature.

**Multidimensional and partial transforms.** In $\mathbb R^d$ the one-dimensional Hilbert transform is replaced by partial Hilbert transforms (applied along a coordinate) or by the Riesz transforms. The Bedrosian identity $T(fg)=fTg$ for partial Hilbert transforms, and its characterization for general singular integrals and Fourier multipliers, is the subject of the multidimensional analytic-signal program (Bulow–Sommer; Lin–Zhang; Venouziou–Zhang). The commutator viewpoint extends verbatim: the identity holds exactly when the projection commutes with multiplication, and the support condition becomes a condition on the sumset of the spectra relative to the jump set of the multiplier symbol. A notable negative result is that the full Riesz transform in $\mathbb R^d$ ($d\ge2$) does **not** satisfy a Bedrosian identity — it is essentially a one-dimensional phenomenon — which is why the partial (coordinate-wise) transforms are the correct multidimensional objects.

**Characterization and necessity theorems.** The modern treatment establishes not only the support condition but *characterizations*: which operators (linear combinations of partial Hilbert transforms and the identity) satisfy the Bedrosian property, and necessity results showing that the support condition cannot be weakened. These results (Venouziou–Zhang; Lin–Zhang; Wen) are the analytical backbone that turns the engineering "rule" into a theorem with a sharp boundary.

---

## 8. Conclusion

Bedrosian's theorem is best understood not as a product rule for the Hilbert transform but as a commutation theorem: $H(fg)=fH(g)$ holds exactly when the Riesz projection $P=\frac12(I+iH)$ commutes with multiplication by the low-pass factor, which in turn holds exactly when the sumset of the two spectra avoids $0$ in both directions. The "non-overlapping spectra" hypothesis is therefore not a convenient sufficient condition but a sharp, necessary and sufficient one. From this vantage the theorem unifies a great deal: it is the multiplicative twin of the Kramers–Kronig duality, the local justification of instantaneous frequency and the Hilbert–Huang transform, the reason the FFT-based analytic-signal pipeline works (and the diagnostic for when it fails), and the seed of the multidimensional analytic-signal program. The Hilbert transform, met through the analytic-signal door, turns out to be governed by the same half-plane analyticity that governs causality — and Bedrosian's theorem is the statement that multiplication by a band-limited factor respects that analyticity precisely when the spectra do not interfere.

---

## References

1. **E. Bedrosian**, "A Product Theorem for Hilbert Transforms," *Proceedings of the IEEE*, vol. 51, no. 5, pp. 868–869, May 1963. (The theorem proper.)
2. **J. L. Brown**, "Analytic Signals and Product Theorems for Hilbert Transforms," *IEEE Transactions on Circuits and Systems*, vol. CAS-21, no. 4, pp. 790–792, Nov. 1974.
3. **J. L. Brown**, "A Hilbert Transform Product Theorem," *Proceedings of the IEEE*, vol. 74, no. 4, pp. 520–521, Apr. 1986.
4. **S. L. Marple, Jr.**, "Computing the Discrete-Time 'Analytic' Signal via FFT," *IEEE Transactions on Signal Processing*, vol. 47, no. 9, pp. 2600–2603, Sep. 1999.
5. **R. Lin, H. Zhang**, "Existence of the Bedrosian Identity for Singular Integral Operators," arXiv:1407.0861 [math.CA], 2014.
6. **R. Lin, H. Zhang**, "Existence of the Bedrosian Identity for Fourier Multiplier Operators," *Forum Mathematicum*, vol. 28, no. 4, pp. 749–759, 2016. (Sharp support condition; necessity.)
7. **M. Venouziou, H. Zhang**, "Characterizing the Hilbert Transform by the Bedrosian Theorem," *Journal of Mathematical Analysis and Applications*, vol. 338, pp. 1477–1481, 2008.
8. **B. Yu, H. Zhang**, "The Bedrosian Identity and Homogeneous Semi-Convolution Equations," *Journal of Integral Equations and Applications*, vol. 20, no. 3, pp. 527–568, 2008.
9. **H. Zhang**, "Multidimensional Analytic Signals and the Bedrosian Identity," *Integral Equations and Operator Theory*, vol. 78, no. 3, 2014; arXiv:1212.6602.
10. **Z. Wen, G. Deng**, "The Bedrosian Identity for $L^p$ Functions and the Hardy Space on Tube," *AIMS Mathematics*, vol. 1, no. 1, pp. 9–23, 2016.
11. **Z. Wen**, "The Bedrosian Identity for Generalized Functions," *Mathematical Methods in the Applied Sciences*, 2021, doi:10.1002/mma.6964.
12. **T. Bülow, G. Sommer**, "Hypercomplex Signals — A Novel Extension of the Analytic Signal to the Multidimensional Case," *IEEE Transactions on Signal Processing*, vol. 49, no. 11, 2001.
13. **N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q. Zheng, N.-C. Yen, C. C. Tung, H. H. Liu**, "The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and Non-Stationary Time Series Analysis," *Proceedings of the Royal Society of London A*, vol. 454, no. 1971, pp. 903–996, 1998.
14. **E. M. Stein**, *Singular Integrals and Differentiability Properties of Functions*, Princeton University Press, 1970.
15. **R. de L. Kronig**, "On the Theory of the Dispersion of X-Rays," *Journal of the Optical Society of America*, vol. 12, no. 6, pp. 547–557, 1926; and **H. A. Kramers**, "La Diffusion de la Lumière par les Atomes," *Atti del Congresso Internazionale dei Fisici*, Como, vol. 2, pp. 545–557, 1927. (The Kramers–Kronig relations, via the Sokhotski–Plemelj theorem.)
