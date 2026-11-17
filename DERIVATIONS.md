# One leak-rate distribution, two readouts: hyperbolic discounting and the place-field width distribution

The multi-scale place-field readout (Figure 3, panels A–C) and the discounting
readout (Figure 3, panels D–E) are two readouts of a **single** population-level object: the
distribution of leak rates $p(\lambda)$ across the bank of leaky integrators
(Eqs. 15–16 of the note). This document derives, from first principles,

1. the **temporal** readout — the discount function $D(t)$ is the Laplace
   transform of $p(\lambda)$, and an exponential $p(\lambda)$ gives exactly
   **hyperbolic** discounting; and
2. the **spatial** readout — place-field width scales as $w\propto\lambda^{-1/2}$,
   so a leak-rate distribution induces a width distribution by pushforward.

Two cases matter. The clean idealization (§3.3–3.7) takes $p(\lambda)$
exponential, giving *exactly* hyperbolic discounting and **Fréchet-2** widths.
The empirical case (§3.8) uses what CA1 actually measures — the *time constants*
$\tau=1/\lambda$ are exponential — giving **Rayleigh** widths that match the
aggregated data (Fig. 3C) and *approximately* hyperbolic discounting (Fig. 3D).
Throughout, $\lambda\ge 0$ is a leak rate and the exponential idealization is

$$
p(\lambda)=\frac{1}{\mu}\,e^{-\lambda/\mu},\qquad \mathbb{E}[\lambda]=\mu .
$$

*This file has three self-contained parts. The one above (§§1–5) is the leak-rate
story for Figure 3. Then [**Context differentiation with training
(Figure 5)**](#context-differentiation-with-training-figure-5) (§§B.1–B.3) derives
the remapping mechanism and gives an ideal-observer justification for its one
assumption. Finally [**Antithetic sampling of theta sweeps
(Figure 1)**](#antithetic-sampling-of-theta-sweeps-figure-1) (§§C.1–C.9) derives
the left-right-alternating sweep construction, its unbiasedness, a closed form for
exactly how much Monte-Carlo variance it removes, the condition under which the
microscopic sampling stays faithful to the macroscopic master equation, and a
depleting-reservoir mechanism that is provably faithful to the generator while
being antithetic by construction, two circuits that could run it, and a closed-form
analysis of the theta-skipping pacemaker with phase diffusion.*

---

## 1. Setup: the two roles of a leak rate

A single CA1 channel is a leaky integrator (Eq. 2),
$\tau\,\dot y_\lambda = -\lambda\, y_\lambda + z$, driven by the CA3 state $z$.
Two consequences of the leak rate $\lambda$ matter here.

**Temporal.** At its fixed point the channel computes the resolvent
$y_\lambda(\infty)=x(\lambda I - W)^{-1}$ (Eq. 8), and the value of a reward
arriving at delay $t$ is discounted by the kernel $e^{-\lambda t}$ (Eq. 13):

$$
V_\lambda=\int_0^\infty e^{-\lambda t}\,r_t\,dt .
$$

So a single leak rate $\lambda$ implements **exponential** temporal discounting
with rate $\lambda$.

**Spatial.** For 2-D exploration, $W$ is (the generator of) a diffusion, and the
resolvent $(\lambda I-W)^{-1}$ is a screened Green's function whose spatial
extent — the place-field width — is set by $\lambda$ (derived in §3.1–3.2):

$$
w \;=\; \frac{c}{\sqrt{\lambda}},\qquad c=2\sqrt{D},
$$

with $D$ the diffusion coefficient.

A population with a *distribution* of leak rates therefore produces a
distribution of discount timescales **and** a distribution of field widths from
the same $p(\lambda)$.

---

## 2. Temporal readout: hyperbolic discounting

### 2.1 The bank discounts by the Laplace transform of $p(\lambda)$

Averaging the single-channel kernel $e^{-\lambda t}$ over the population gives the
effective discount function

$$
D(t)\;=\;\mathbb{E}_{\lambda}\!\left[e^{-\lambda t}\right]
\;=\;\int_0^\infty p(\lambda)\,e^{-\lambda t}\,d\lambda
\;=\;\widehat{p}(t),
$$

i.e. $D$ is exactly the **Laplace transform** of $p$ evaluated at the delay $t$.

### 2.2 Exponential leak rates $\Rightarrow$ hyperbolic discounting

Insert $p(\lambda)=\tfrac1\mu e^{-\lambda/\mu}$:

$$
D(t)=\int_0^\infty \frac{1}{\mu}\,e^{-\lambda/\mu}\,e^{-\lambda t}\,d\lambda
=\frac{1}{\mu}\int_0^\infty e^{-\lambda\left(\frac1\mu+t\right)}\,d\lambda
=\frac{1}{\mu}\cdot\frac{1}{\frac1\mu+t},
$$

using $\int_0^\infty e^{-a\lambda}d\lambda=1/a$ for $a=\tfrac1\mu+t>0$. Hence

$$
\boxed{\,D(t)=\dfrac{1}{1+\mu t}\,}
$$

— **hyperbolic discounting** (Mazur's form, with discount parameter $k=\mu$).
Note $D(0)=1$ and the tail $D(t)\sim 1/(\mu t)$ is far heavier than any single
exponential $e^{-\lambda t}$.

### 2.3 Uniqueness

The map $p\mapsto D$ is the Laplace transform, which is injective. Hence
$D(t)=1/(1+\mu t)$ holds **iff** $p$ is its inverse Laplace transform. Writing
$\tfrac{1}{1+\mu t}=\tfrac1\mu\cdot\tfrac{1}{t+1/\mu}$ and using
$\mathcal L\{e^{-a\lambda}\}(t)=\tfrac{1}{t+a}$,

$$
p(\lambda)=\mathcal L^{-1}\!\left\{\tfrac{1}{1+\mu t}\right\}(\lambda)
=\frac1\mu\,e^{-\lambda/\mu}.
$$

So the exponential is the **unique** leak-rate distribution giving exactly
hyperbolic discounting. This pins $p(\lambda)$.

### 2.4 Generalization (Gamma $\Rightarrow$ generalized hyperbola)

For a Gamma prior $p(\lambda)=\dfrac{\beta^{\alpha}}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda}$,

$$
D(t)=\int_0^\infty p(\lambda)\,e^{-\lambda t}\,d\lambda
=\left(\frac{\beta}{\beta+t}\right)^{\!\alpha}
=\frac{1}{\left(1+t/\beta\right)^{\alpha}} ,
$$

the **generalized hyperbola**. The exponential is the special case
$\alpha=1,\ \beta=1/\mu$, recovering $1/(1+\mu t)$; the constant-rate exponential
discount $e^{-\bar\lambda t}$ is the degenerate limit $\alpha\to\infty$.

This Gamma-mixture discount function recurs across fields, always as the Laplace
transform of a Gamma distribution of rates:

- **Behavioral economics / psychology.** Loewenstein & Prelec (1992) introduced
  exactly this form, $D(t)=(1+\alpha t)^{-\gamma/\alpha}$ (our
  $\alpha_{\text{here}}=\gamma/\alpha$, $\beta=1/\alpha$); Mazur's (1987) simple
  hyperbola $1/(1+kt)$ is the $\alpha=1$ (exponential-rate) case.
- **Biology.** Sozou (1998) derived it as rational discounting under an
  *uncertain hazard rate*: an exponential prior on the hazard gives the simple
  hyperbola, a Gamma/Erlang prior the generalized hyperbola — the same
  rate-mixing as here.
- **Economics of the far future.** Weitzman (2001), "Gamma discounting," puts a
  Gamma prior directly on the discount rate, yielding a declining effective
  rate; Farmer & Geanakoplos (2009) make the same point for valuing the far
  future under discount-rate uncertainty.
- **Reinforcement learning / neuroscience.** Fedus et al. (2019) write hyperbolic
  discounting as $d_t=\int_0^1\gamma^t w(\gamma)\,d\gamma$ — a mixture of
  exponential discounts learned over multiple horizons — and multi-timescale
  value codes (Momennejad & Howard 2018; Tano et al. 2020; Masset et al. 2025,
  cited in the note) realize the same bank-of-rates idea in the brain.

Here the novelty is not the discount function itself but that the *same* rate
distribution is also read out spatially as place-field widths (§3).

---

## 3. Spatial readout: the Fréchet width distribution

### 3.1 From the finite-state resolvent to the continuum Green's function

The model is finite-state: $W$ is the generator of the CA3 diffusive walk on the
discretized arena (the `open_arena` lattice). The place field of the cell coding
location $j$ is the $j$-th column of the resolvent,
$g_j \equiv R[:,j]=(\lambda I-W)^{-1}e_j$. Because $(\lambda I-W)R=I$, this column
solves a **discrete screened-Poisson equation** with a unit point source at $j$,

$$
(\lambda I-W)\,g_j=e_j ,
$$

so $g_j$ is the lattice Green's function of $\lambda I-W$. We now show its
continuum limit is the screened-Poisson Green's function.

**(i) The generator is a graph Laplacian.** For the 2-D nearest-neighbour walk
with hop rate $k$ along each edge and lattice spacing $h$, the off-diagonal rates
are $W_{il}=k$ for neighbours and $W_{ii}=-(\deg i)\,k$, so $W$ acts on a field as
$k$ times the graph Laplacian,

$$
(Wg)_i=k\sum_{l\sim i}\big(g_l-g_i\big).
$$

**(ii) Continuum operator.** Taylor-expanding the 4-neighbour stencil,
$\sum_{l\sim i}\big(g_l-g_i\big)=h^2\nabla^2 g(\mathbf x_i)+O(h^4)$, so

$$
Wg \;\approx\; k h^2\,\nabla^2 g \;=\; D\,\nabla^2 g,
\qquad D=k h^2,
$$

with $D$ the diffusion coefficient (equivalently $\langle r^2\rangle=4Dt$).

**(iii) Diffusive-scaling limit.** Take $h\to0,\ k\to\infty$ with $D=kh^2$ held
fixed, and rescale the source $e_j/h^2\to\delta(\mathbf x-\mathbf x_j)$ (a unit
mass on one site becomes a unit-integral density once divided by the cell area
$h^2$). The discrete equation
$\lambda g_i-k\sum_{l\sim i}(g_l-g_i)=(e_j)_i$ becomes the **continuum
screened-Poisson (Yukawa) equation**

$$
\big(\lambda-D\nabla^2\big)\,G(\mathbf r)=\delta(\mathbf r),
\qquad g_j\to G .
$$

**(iv) Solve by Fourier transform.** Plane waves diagonalize $-D\nabla^2$ with
eigenvalue $D|\mathbf q|^2$ — the continuum image of the spectral decomposition
of $W$ (Eqs. 6–7) — so

$$
G(\mathbf r)=\int\!\frac{d^2q}{(2\pi)^2}\,
\frac{e^{i\mathbf q\cdot\mathbf r}}{\lambda+D|\mathbf q|^2}
=\frac{1}{2\pi D}\,K_0\!\Big(\frac{r}{\xi}\Big),
\qquad \xi=\sqrt{\frac{D}{\lambda}} ,
$$

a modified Bessel function with screening length $\xi\propto\lambda^{-1/2}$
(equivalently, the radial Yukawa solution decaying at infinity, since
$K_0(r/\xi)\sim e^{-r/\xi}$ for $r\gg\xi$).

**(v) Regime of validity.** The continuum form is accurate when the screening
length exceeds the lattice spacing, $\xi\gg h$ — i.e. for small $\lambda$ (large
fields). As $\lambda$ grows, $\xi\to h$ and the field collapses onto a single
site; lattice discreteness then **floors** the width at $w_{\min}\sim h$ — the
floor invoked in §3.6. (In the simulation $h=1$, $k=1$, so $D=1$.)

### 3.2 Field width: $w=c/\sqrt\lambda$

Measuring width as the RMS radius of $G(r)\propto K_0(r/\xi)$,

$$
w^2=\frac{\displaystyle\int_0^\infty r^2\,G(r)\,2\pi r\,dr}{\displaystyle\int_0^\infty G(r)\,2\pi r\,dr}
=\xi^2\,\frac{\displaystyle\int_0^\infty x^3 K_0(x)\,dx}{\displaystyle\int_0^\infty x\,K_0(x)\,dx}
\qquad(x=r/\xi).
$$

Using $\int_0^\infty x^{\mu-1}K_0(x)\,dx=2^{\mu-2}\,\Gamma(\mu/2)^2$,

$$
\int_0^\infty x\,K_0\,dx=2^{0}\Gamma(1)^2=1,\qquad
\int_0^\infty x^3 K_0\,dx=2^{2}\Gamma(2)^2=4,
$$

so $w^2=4\xi^2$ and

$$
\boxed{\,w=2\xi=2\sqrt{D/\lambda}=\dfrac{c}{\sqrt\lambda},\quad c=2\sqrt D\,}
\qquad\Longleftrightarrow\qquad \lambda=\frac{c^2}{w^2}.
$$

Any other width measure (FWHM, space constant) is also $\propto\xi$, changing
only $c$. The continuum estimate $c=2\sqrt D=2$ (with $D=1$) is close to the
$c\approx1.8$ read off the measured $w(\lambda)$ curve (Figure 3B); the small
gap is the lattice / finite-arena correction to the idealized continuum.

### 3.3 Pushforward of $p(\lambda)$ through $w=c/\sqrt\lambda$

**CDF method (cleanest).** $w=c/\sqrt\lambda$ is decreasing in $\lambda$, so

$$
F_w(x)=\Pr(w\le x)=\Pr\!\left(\lambda\ge \frac{c^2}{x^2}\right)
=e^{-\frac{1}{\mu}\frac{c^2}{x^2}}
=e^{-a/x^2},\qquad a\equiv\frac{c^2}{\mu},
$$

using $\Pr(\lambda\ge \ell)=e^{-\ell/\mu}$ for the exponential. Differentiating,

$$
\boxed{\,p_w(w)=\dfrac{dF_w}{dw}=\dfrac{2a}{w^{3}}\,e^{-a/w^{2}},\quad w>0\,}.
$$

**Jacobian check.** With $\lambda(w)=c^2/w^2$ and $|d\lambda/dw|=2c^2/w^3$,

$$
p_w(w)=p_\lambda(\lambda(w))\left|\frac{d\lambda}{dw}\right|
=\frac1\mu e^{-c^2/(\mu w^2)}\cdot\frac{2c^2}{w^3}
=\frac{2a}{w^3}e^{-a/w^2}.\ \checkmark
$$

### 3.4 This is a Fréchet (inverse-Weibull) distribution

The Fréchet density with shape $\alpha$ and scale $s$ is
$f(w)=\alpha\,s^{\alpha}w^{-(\alpha+1)}e^{-(s/w)^{\alpha}}$, with CDF
$F(w)=e^{-(s/w)^{\alpha}}$. Matching $F_w(x)=e^{-a/x^2}$ gives

$$
\boxed{\,\alpha=2,\qquad s=\sqrt a=\dfrac{c}{\sqrt\mu}\,}
$$

so the widths are **Fréchet$(\alpha=2,\ s=c/\sqrt\mu)$**. Equivalently,
$w^{-2}=\lambda/c^2$ is itself exponential, which is the defining property of a
Fréchet-2 variate.

### 3.5 Properties

From $p_w(w)=2a\,w^{-3}e^{-a/w^2}$ and the Fréchet moment formula
$\mathbb E[w^k]=s^k\,\Gamma(1-k/\alpha)$ (finite only for $k<\alpha=2$):

| quantity | expression | value at $c=1.8,\ \mu=1$ | simulation |
|---|---|---|---|
| mode | $\sqrt{2a/3}=\sqrt{2/3}\,(c/\sqrt\mu)$ | $1.47$ | peak near floor |
| median | $\sqrt{a/\ln 2}=1.20\,(c/\sqrt\mu)$ | $2.16$ | $2.4$ |
| mean | $\sqrt a\,\Gamma(\tfrac12)=\sqrt{\pi}\,(c/\sqrt\mu)=c\sqrt{\pi/\mu}$ | $3.19$ | $3.1$ |
| variance | $a\,\Gamma(0)=\infty$ | $\infty$ | (cut by boundary) |
| right tail | $p_w(w)\sim 2a\,w^{-3}$ | power law | — |

- **Mode** (set $\tfrac{d}{dw}\ln p_w=-3/w+2a/w^3=0$): $w_{\text{mode}}=\sqrt{2a/3}$.
- **Mean** $=c\sqrt{\pi/\mu}$ decreases with $\mu$: faster discounting ⇒ smaller
  fields. The predicted $3.19$ matches the simulated $3.1$.
- **Infinite variance / $w^{-3}$ tail**: the Fréchet is heavy-tailed. In the
  finite arena the boundary truncates this tail, making the empirical variance
  finite.

### 3.6 When a Fréchet body looks exponential (single sessions)

The Fréchet density rises from $0$, peaks at $w_{\text{mode}}=\sqrt{2a/3}$, then
falls as $w^{-3}$. Two features of the simulation reshape it on the *observable*
window $[w_{\min},w_{\max}]$:

- a **floor** $w_{\min}$ — the smallest resolvable field (≈ one bin / the
  diffusion–discretization length); and
- a **cap** $w_{\max}$ — the arena boundary, which removes the heavy tail.

If $\mu$ is large enough that the mode falls at or below the floor,
$w_{\text{mode}}=\sqrt{2a/3}\le w_{\min}$, i.e.

$$
\boxed{\ \mu\;\ge\;\frac{2}{3}\,\frac{c^2}{w_{\min}^2}\ }
$$

then $p_w$ is **monotonically decreasing** across $[w_{\min},w_{\max}]$, so a
single session or a near-floor regime can look roughly exponential — the sense in
which earlier single-session reports appear exponential. The **aggregated** CA1
widths (Cohen & Drugowitsch 2025, $n=32{,}000$), however, are clearly *unimodal*
(mode $\approx7$ cm), which is the Rayleigh pushforward of the exponential-$\tau$
data rather than a monotone exponential — see §3.8, which is what Figure 3C
plots.

---

### 3.7 Fréchet vs Gamma: can the Fréchet substitute for the empirical fit?

Harland et al. fit a **Gamma** to CA1 tuning widths,
$\sigma\sim\Gamma(v,\beta/v)$, and report a good fit (their Extended Data
Fig. 4f). They chose the Gamma as a convenient, conventional family ("to align
with previous studies"); it is descriptive, not mechanistic. Our model instead
**derives** a Fréchet-2 width law (§3.4). Are the two interchangeable?

**Body: yes.** Both are unimodal right-skewed laws, and over the bulk of the
distribution — where essentially all recorded fields lie — they nearly coincide
([`figures/frechet_vs_gamma.pdf`](figures/frechet_vs_gamma.pdf), left). A *free*
Fréchet fits Gamma-distributed widths as well as a Gamma does, and vice versa:

| width data ($n=10^3$, mean KS) | KS, Gamma fit | KS, Fréchet fit |
|---|---|---|
| Gamma, $v=1.3$ (near-exponential) | $0.063$ | $0.065$ (free) |
| Gamma, $v=2$ | $0.055$ | $0.065$ (free) |
| Fréchet-2 (the model) | $0.095$ | $0.032$ |

(reproduced by [`analysis_frechet_vs_gamma.py`](analysis_frechet_vs_gamma.py))

**Tail: they differ.** The Fréchet-2 tail is the power law $p_w\sim 2a\,w^{-3}$
(infinite variance, §3.5); the Gamma tail is exponential. On a log axis the two
fits separate at large $w$ (`frechet_vs_gamma.pdf`, right). This is the only
place they are distinguishable, and it is exactly where the data are weakest:
the finite arena truncates the tail and a session contributes few very-large
fields. A Gamma fit to truly Fréchet-2 widths still gives only $\mathrm{KS}\approx0.10$
— visually "a good fit", consistent with the paper's report.

**Verdict.** Over the empirically constrained range the Fréchet substitutes for
the Gamma: the paper's good Gamma fit does **not** exclude the mechanistic
Fréchet, and adopting the Fréchet buys the unified picture (one exponential
$p(\lambda)\Rightarrow$ hyperbolic discounting *and* the place-field widths). The
one extra, testable commitment is a heavier ($w^{-3}$) tail — resolvable only
with large arenas and many cells.

**If a lighter tail is needed.** The framework interpolates. Take a *Gamma*
leak-rate distribution $p(\lambda)=\Gamma(\kappa,\beta)$ (the exponential is
$\kappa=1$). Then discounting is the **generalized hyperbola**

$$
D(t)=\Big(\tfrac{\beta}{\beta+t}\Big)^{\kappa}=\big(1+t/\beta\big)^{-\kappa},
$$

and the width law (pushforward through $w=c/\sqrt\lambda$) is

$$
p_w(w)=\frac{2\beta^{\kappa}c^{2\kappa}}{\Gamma(\kappa)}\,
w^{-(2\kappa+1)}\,e^{-\beta c^2/w^2},
$$

with tail $w^{-(2\kappa+1)}$ that *lightens* as $\kappa$ grows. So a small
$\kappa>1$ trades exact hyperbolicity for slight ($\propto t^{-\kappa}$)
near-hyperbolic discounting and a lighter, more Gamma-like width tail. The
exponential / Fréchet-2 case ($\kappa=1$) is the exactly-hyperbolic limit used in
the figures.

---

### 3.8 The empirical case: exponential *time constants* ⇒ Rayleigh widths

§3.3–3.4 assumed the *leak rate* $\lambda$ is exponential. The CA1 data measure
the *time constant* $\tau=1/\lambda$, and it is $\tau$ that is exponential
(doi:10.1002/hipo.70105; KS $p=0.91$). The Fig. 3C inset shows the leak-rate
distribution $\lambda=1/\tau$ with its **inverse-exponential** fit (the density of
$1/\tau$ for $\tau\sim\mathrm{Exp}$); pushing exactly that distribution through the
width relation gives the Rayleigh curve in the main panel, so the inset and the
prediction are strictly one law. With
$\tau\sim\mathrm{Exp}(\mu_\tau)$ and
$w=c/\sqrt\lambda=c\sqrt\tau$,

$$
F_w(x)=\Pr(c\sqrt\tau\le x)=\Pr\!\Big(\tau\le\tfrac{x^2}{c^2}\Big)
=1-e^{-x^2/(c^2\mu_\tau)},\qquad
p_w(w)=\frac{2w}{c^2\mu_\tau}\,e^{-w^2/(c^2\mu_\tau)} ,
$$

a **Rayleigh** law (equivalently Weibull with shape $2$), scale
$\sigma=c\sqrt{\mu_\tau/2}$, i.e. $w=c\sqrt\tau$ with $w^2\propto\tau$ exponential.
Unlike the Fréchet it is **unimodal with a light (Gaussian) tail** and finite
moments: mean $\sigma\sqrt{\pi/2}$, median $\sigma\sqrt{2\ln2}$. Calibrating the
single scale $c\approx3.28\ \mathrm{cm\cdot s^{1/2}}$ to the empirical median
($\mu_\tau=\overline\tau\approx13.7\,$s) predicts the whole distribution:

| quantity | prediction (Rayleigh) | empirical ($n=32{,}000$) |
|---|---|---|
| median | $10.1$ cm | $10.1$ cm |
| mean | $10.7$ cm | $10.9$ cm |
| p95 | $21.0$ cm | $22.3$ cm |

So the two readouts share **one** empirical distribution ($\tau\sim\mathrm{Exp}$):
its width pushforward is Rayleigh (matching the aggregated Cohen & Drugowitsch
2025 / Pfeiffer & Foster 2013 widths, Fig. 3C), and its Laplace transform is
*approximately* hyperbolic (Fig. 3D; §4). The exponential-$\lambda$ / Fréchet
account of §3.3–3.7 is the idealization that makes discounting *exactly*
hyperbolic; it predicts a heavier $w^{-3}$ width tail (p95 $\approx37$ cm) than
observed, which is why the exponential-$\tau$ data specifically select the
lighter Rayleigh.

### 3.9 The Rayleigh widths *are* a discount function

The exponential-$\tau$ law of §3.8 is read out temporally as well as spatially,
and the two readouts are the *same distribution* seen through the two links
$w=c/\sqrt\lambda$ and $D=\mathcal L\{p_\lambda\}$ (§2.1).

**Bridge identity.** From $w=c/\sqrt\lambda$, a cell's leak rate and field width
obey
$$
\lambda=\frac{c^2}{w^2},\qquad \tau=\frac1\lambda=\frac{w^2}{c^2},
$$
so a cell's discount time constant equals the time to diffuse across its own
place field ($w^2/c^2$, with $c^2$ the diffusion constant of §3.1). Hence
$w^2=c^2\tau$: the Rayleigh width law ($w^2\sim\mathrm{Exp}$, §3.8) and the
exponential time-constant law are the *same* random variable up to the fixed
scale $c^2$.

**Discount function.** The discount factor is the Laplace transform of the
leak-rate density (§2.1), which we may write as an average over the widths:
$$
D(t)=\mathbb E\!\left[e^{-\lambda t}\right]
=\mathbb E_{w\sim\text{Rayleigh}}\!\left[e^{-c^2 t/w^2}\right]
=\int_0^\infty \frac{1}{\mu_\tau}e^{-\tau/\mu_\tau}\,e^{-t/\tau}\,d\tau .
$$
With the integral $\int_0^\infty e^{-a/x-bx}\,dx=2\sqrt{a/b}\,K_1\!\big(2\sqrt{ab}\big)$
(modified Bessel function of the second kind; here $a=t$, $b=1/\mu_\tau$),
$$
\boxed{\,D(t)=u\,K_1(u),\qquad u=2\sqrt{t/\mu_\tau}\,}
$$
(verified against numerical quadrature to machine precision). Equivalently
$u=(c/\sigma)\sqrt{2t}$ with the Rayleigh scale $\sigma=c\sqrt{\mu_\tau/2}$ (§3.8),
so a single pair $(c,\mu_\tau)$ sets both the spatial spread and the temporal
horizon.

**Behaviour.**
- *Short delays.* $uK_1(u)\to1$ as $t\to0$, so $D(0)=1$, and
$$
D(t)=1+\frac{t}{\mu_\tau}\Big[\ln\tfrac{t}{\mu_\tau}+2\gamma-1\Big]+O(t^2\ln t).
$$
The nonanalytic $t\ln t$ term gives $D'(0^+)=-\infty$ — the signature of
$\mathbb E[\lambda]=\mathbb E[1/\tau]=\infty$ for $\tau\sim\mathrm{Exp}$ (the
density is positive at $\tau=0$, so arbitrarily fast cells produce an arbitrarily
steep initial drop). The empirical distribution is bounded away from $\tau=0$
(smallest measured $\tau\approx1$ s), regularizing the slope to the finite
$-\mathbb E_{\text{emp}}[1/\tau]\approx-0.19\ \mathrm{s^{-1}}$.
- *Long delays.* $K_1(u)\sim\sqrt{\pi/2u}\,e^{-u}$, so
$D(t)\sim\sqrt{\pi}\,(t/\mu_\tau)^{1/4}\,e^{-2\sqrt{t/\mu_\tau}}$ — a **stretched
exponential**: slower than any single exponential, faster than the power-law
hyperbolic $1/(\mu t)$.

**Relationship to hyperbolic discounting.** Exactly hyperbolic discounting
$D(t)=1/(1+\mu t)$ requires $\lambda\sim\mathrm{Exp}$ (§2.2), whose width
pushforward is Fréchet-2 (§3.4), *not* Rayleigh. The Rayleigh-generating
$\tau\sim\mathrm{Exp}$ instead gives the Bessel form $uK_1(u)$, which is
**approximately hyperbolic**: over behaviourally relevant delays it stays within
$\approx0.03$ of $1/(1+\mu t)$ (taking $\mu=\mathbb E_{\text{emp}}[1/\tau]$;
Fig. 3D), sharing every qualitative property that matters — monotone, convex,
far more sustained than a single timescale, hence producing preference reversals
(Fig. 3E) — and differing only in the deep tail (stretched-exponential vs
power-law). In one line,
$$
\underbrace{w\sim\text{Rayleigh}}_{\text{spatial}}
\quad\Longleftrightarrow\quad
\underbrace{D(t)=2\sqrt{t/\mu_\tau}\;K_1\!\big(2\sqrt{t/\mu_\tau}\big)\approx\frac{1}{1+\mu t}}_{\text{temporal}} ,
$$
one distribution ($w^2=c^2\tau$), two readouts. The clean idealization that makes
discounting *exactly* hyperbolic ($\lambda\sim\mathrm{Exp}$, at the cost of
Fréchet rather than Rayleigh widths) is the subject of §4.

---

## 4. The two exact forms are incompatible

The change of variables $w=c/\sqrt\lambda$ and the Laplace transform $p\mapsto D$
are both bijections, so $p(\lambda)$ cannot be exponential in $\lambda$ **and**
yield an exponential in $w$ simultaneously:

- **Hyperbolic discounting** forces $p(\lambda)\propto e^{-\lambda/\mu}$ (§2.3),
  whose width pushforward is Fréchet-2 (§3.4) — *not* exponential.
- **Exactly exponential widths** $p_w(w)=\tfrac1s e^{-w/s}$ would require, by the
  inverse pushforward,

$$
p_\lambda(\lambda)=p_w\!\big(w(\lambda)\big)\left|\frac{dw}{d\lambda}\right|
=\frac{c}{2s}\,\lambda^{-3/2}\,e^{-(c/s)\lambda^{-1/2}},
$$

  which is heavy-tailed in $\lambda$ and whose Laplace transform is **not**
  $1/(1+\mu t)$.

Which one to make exact is an empirical question, and the CA1 data answer it: the
measured *time constants* $\tau=1/\lambda$ are exponential, and the aggregated
tuning widths match the Rayleigh pushforward of that distribution (§3.8). So the
figures anchor on **exponential $\tau$** — exact Rayleigh widths (matching the
data, Fig. 3C) and *approximately* hyperbolic discounting (max abs. deviation
$0.033$ from $1/(1+\mu t)$, Fig. 3D). Anchoring instead on **exponential
$\lambda$** — the note's exactly-hyperbolic idealization — gives the heavier-
tailed Fréchet-2 widths (§3.4). The two agree over the body and differ only in
the (weakly sampled) large-width tail; the exponential-$\tau$ data favour the
lighter Rayleigh.

---

## 5. Summary

A single population-level leak-rate distribution drives both readouts, linked by
$D=\mathcal L\{p_\lambda\}$ and $w=c/\sqrt\lambda$. The CA1 data fix that
distribution: the time constants $\tau=1/\lambda$ are exponential
($\mu_\tau=\overline\tau\approx13.7$ s; $\mu=\overline{1/\tau}\approx0.19$
s$^{-1}$), so the figures use

$$
\underbrace{D(t)=2\sqrt{t/\mu_\tau}\,K_1\!\big(2\sqrt{t/\mu_\tau}\big)\approx\frac{1}{1+\mu t}}_{\text{temporal: }\approx\text{ hyperbolic (§3.9, Fig. 3D-E)}}
\qquad\text{and}\qquad
\underbrace{p_w(w)=\frac{2w}{c^2\mu_\tau}e^{-w^2/(c^2\mu_\tau)}}_{\text{spatial: Rayleigh, matches Fig. 3C}}
$$

The exponential-$\lambda$ idealization instead makes discounting *exactly*
hyperbolic $1/(1+\mu t)$ and the widths Fréchet-2 $\tfrac{2a}{w^3}e^{-a/w^2}$
($a=c^2/\mu$) — a heavier tail than observed (§3.8, §4).

| | distribution / function | role | figure | code |
|---|---|---|---|---|
| leak rates | $\lambda=1/\tau,\ \tau\sim\mathrm{Exp}$ | shared cause | Fig. 3C inset | `1/CA1_TAU` |
| discounting | $D(t)\approx1/(1+\mu t)$ | temporal readout | Fig. 3D–E | `sim3_multiscale_discounting.py` |
| widths | Rayleigh$(\sigma=c\sqrt{\mu_\tau/2})$ | spatial readout | Fig. 3A–C | `sim3_multiscale_discounting.py` |

---

# Context differentiation with training (Figure 5)

Figure 5 contrasts **partial/gradual** remapping across a square→circle morph
(Leutgeb et al. 2005) with **complete/abrupt** remapping in the same paradigm
(Wills et al. 2005). Following **Sanders, Wilson & Gershman (2020)**, the
Leutgeb/Wills difference is a property of the training **data** — chiefly *how
much* training preceded the morph test — and **not** a free model parameter. We
capture this with a single, minimal assumption and no added machinery: the two
stored context representations become increasingly **differentiated** with
training. §B.1–B.2 derive the mechanism and show that one differentiation
variable sets both signatures of remapping; §B.3 gives the (optional)
ideal-observer justification for why differentiation grows with training.
Throughout, $\operatorname{logistic}(z)=1/(1+e^{-z})$.

## B.1 The differentiation mechanism (the model)

Let $g\ge 0$ be the **context differentiation** — how separated the two stored
context representations (square, circle) are — and assume it grows monotonically
with the amount of training. A morph cue $m\in[0,1]$ (square at $m=0$, circle at
$m=1$) is assigned to the two contexts by similarity, and the resulting DG
context vector is

$$
\boxed{\;s(m)=\operatorname{logistic}\!\big(g\,(2m-1)\big),\qquad
\mathbf c(m)=\big[\,1-s(m),\ s(m)\,\big].\;}
$$

As the representations differentiate (larger $g$), the *same* morph axis spans a
larger separation, so the assignment $s(m)$ — and hence the remapping — sharpens
on its own, with no separate temperature or inference parameter. The context
$\mathbf c(m)$ enters the network exactly as in Simulation 4, $b=\mathbf c G$,
$J_{ij}=W_{ij}b_j$ (Eqs. 17–18), and the CA1 read-out is the context-gated
truncated resolvent. The two DG ensembles overlap (a fraction of cells are active
in both — the gate `frac`), leaving a shared, non-remapping core.

## B.2 One variable sets both the degree and the abruptness

Two features of $s(m)$ govern the morph response, and a single $g$ controls both.

**Degree of remapping** = separation of the two endpoints,

$$
|s(1)-s(0)|=\operatorname{logistic}(g)-\operatorname{logistic}(-g)=\tanh\!\big(\tfrac g2\big),
$$

rising from $0$ (no remapping) to $1$ (complete) as $g$ grows.

**Abruptness** = slope of the assignment at the midpoint. Since
$s'(m)=2g\,s(m)\big(1-s(m)\big)$ and $s(\tfrac12)=\tfrac12$,

$$
s'\!\big(\tfrac12\big)=2g\cdot\tfrac12\cdot\tfrac12=\frac g2,
$$

also increasing with $g$. Hence the two limits:

- **Low differentiation**, $g\to0$: $s(m)\to\tfrac12$ for every $m$, so
  $\mathbf c(m)\to[\tfrac12,\tfrac12]$ is *morph-independent*; both ensembles are
  driven equally and the CA1 map barely changes ⇒ **gradual, partial remapping
  (matching Leutgeb)**.
- **High differentiation**, $g$ large: $s(m)$ approaches a step at $m=\tfrac12$,
  so $\mathbf c$ switches from $[1,0]$ to $[0,1]$ across a narrow band ⇒ **abrupt,
  complete remapping (matching Wills)**.

Because the ensembles overlap, even a fully confident switch leaves the shared
core unchanged: remapping is *partial at the cell level* (Fig. 5C) even when
complete at the population level. In one line,

$$
\text{training}\;\longrightarrow\;g\;\longrightarrow\;\Big(\underbrace{\tanh(g/2)}_{\text{degree}},\ \underbrace{g/2}_{\text{abruptness}}\Big)\;\longrightarrow\;\text{Leutgeb (small }g)\ \text{vs.\ Wills (large }g).
$$

| object | formula | grows with $g$ | code |
|---|---|---|---|
| soft assignment | $s(m)=\operatorname{logistic}(g(2m-1))$ | — | `context_vector` |
| remapping degree | $\tanh(g/2)$ | $0\to1$ | Fig. 5B |
| transition slope | $g/2$ | $0\to\infty$ | Fig. 5A |

## B.3 Why should differentiation grow with training? An ideal-observer justification

The model simply *posits* that $g$ increases with training. That assumption is
not arbitrary: it is what an ideal observer performing **latent-state inference**
(Sanders et al. 2020; Anderson 1991) would do — with more experience it becomes
confident the square and circle are *distinct* contexts and separates their
representations. This section sketches that normative account. **It is not part of
the model** (the network needs no inference, only the monotone $g$); it explains
*why the assumption is reasonable*. Symbols here are local to §B.3: $\mu,\sigma,\tau,\alpha$
below are unrelated to the leak rate, width scale, or time constant of Part I.

### The generative model (a CRP mixture)

Each environment *presentation* is summarized by a scalar feature $x\in\mathbb R$
(a shape descriptor along the square→circle axis). The two prototypes sit at
$x=-\mu$ (square) and $x=+\mu$ (circle). The rational-analysis generative model
(Anderson 1991; the Chinese-restaurant-process / DP mixture used by Sanders et
al. 2020) is:

- a **partition** of the observations into latent states (clusters) is drawn from
  a Chinese restaurant process, $\text{CRP}(\alpha)$;
- each state $k$ has a mean $\theta_k\sim\mathcal N(0,\tau^2)$;
- an observation assigned to state $k$ is $x\sim\mathcal N(\theta_k,\sigma^2)$
  with **known** variance $\sigma^2$.

Parameters (matching the values that made Fig. 5 concrete): feature mean
$\mu=1$, observation noise $\sigma=0.7$, prior mean-scale $\tau=2$, concentration
$\alpha=0.05$. A small $\alpha$ is a **simplicity bias**: a priori the observer
expects *few* states, and it takes positive evidence to justify more.

### Cluster marginal likelihood (Normal–Normal conjugacy)

For a set of $m$ observations $\mathbf x=\{x_1,\dots,x_m\}$ assigned to a single
state, the state mean $\theta$ is integrated out under its conjugate prior:

$$
p(\mathbf x)=\int_{-\infty}^{\infty}\mathcal N(\theta;0,\tau^2)\prod_{i=1}^{m}\mathcal N(x_i;\theta,\sigma^2)\,d\theta .
$$

Write the prior precision $\rho_0=\tau^{-2}$ and the posterior precision after $m$
points $\rho_m=\rho_0+m/\sigma^2$, with posterior mean
$\bar\theta_m=\big(\sum_i x_i/\sigma^2\big)/\rho_m$ (the prior mean is $0$).
Completing the square in $\theta$ inside the exponent,

$$
\frac1{\sigma^2}\sum_i(x_i-\theta)^2+\rho_0\theta^2
=\frac{\sum_i x_i^2}{\sigma^2}+\rho_m(\theta-\bar\theta_m)^2-\rho_m\bar\theta_m^2 ,
$$

and doing the Gaussian integral $\int e^{-\frac12\rho_m(\theta-\bar\theta_m)^2}d\theta=\sqrt{2\pi/\rho_m}$
against the prior normalizer $\sqrt{\rho_0/2\pi}$ and the likelihood normalizer
$(2\pi\sigma^2)^{-m/2}$ gives

$$
\boxed{\;\log p(\mathbf x)=-\frac m2\log\!\big(2\pi\sigma^2\big)+\frac12\log\frac{\rho_0}{\rho_m}-\frac12\!\left(\frac{\sum_i x_i^2}{\sigma^2}-\rho_m\bar\theta_m^{\,2}\right).\;}
$$

The middle term $\tfrac12\log(\rho_0/\rho_m)<0$ is the **Occam factor** — the
likelihood penalty a model pays for having an adjustable mean.

### One state vs. two: the posterior odds

The training protocol alternates square and circle presentations, so after $N$
observations there are $N/2$ of each, drawn $x_i\sim\mathcal N(\mp\mu,\sigma^2)$.
Two partitions dominate the CRP posterior and are the ones compared:

- $H_1$: **one** state — all $N$ observations in a single cluster;
- $H_2$: **two** states — squares in one cluster, circles in the other.

($H_2$ is the maximum-likelihood partition and $H_1$ the maximum-prior partition,
so their contrast captures the whole Occam trade-off; the remaining $B_N-2$
partitions carry negligible mass for a cleanly bimodal stimulus set, so comparing
just these two keeps the argument tractable.)

**Prior ratio.** The CRP assigns a partition of $N$ items into $K$ blocks of
sizes $n_k$ the exchangeable probability

$$
P(\text{partition})=\frac{\alpha^{K}\prod_{k=1}^{K}(n_k-1)!}{\prod_{i=0}^{N-1}(\alpha+i)}
=\frac{\Gamma(\alpha)}{\Gamma(\alpha+N)}\,\alpha^{K}\prod_{k=1}^{K}\Gamma(n_k).
$$

For $H_1$ ($K=1,\ n_1=N$) and $H_2$ ($K=2,\ n_1=n_2=N/2$) the shared factor
$\Gamma(\alpha)/\Gamma(\alpha+N)$ cancels, leaving

$$
\log\frac{P(H_2)}{P(H_1)}=\log\alpha+2\log\Gamma\!\Big(\frac N2\Big)-\log\Gamma(N),
$$

the prior log-odds of two states.

**Posterior log-odds.** Because observations are independent given the partition,
the marginal likelihood of $H_2$ factorizes over its two clusters. Hence the
log-odds in favour of two states is the prior ratio plus the log Bayes factor,

$$
d=\underbrace{\log\alpha+2\log\Gamma\!\big(\tfrac N2\big)-\log\Gamma(N)}_{\text{prior (Occam)}}
+\underbrace{\log p(\mathbf x_{\text{sq}})+\log p(\mathbf x_{\text{ci}})-\log p(\mathbf x_{\text{all}})}_{\text{evidence for splitting}},
$$

and the posterior probability of two states is
$P(H_2\mid\mathbf x)=\operatorname{logistic}(d)$. The reported curve is the
**expected** posterior over training draws,

$$
\boxed{\;P(2\text{ states}\mid N)=\mathbb E_{\text{train}}\big[\operatorname{logistic}(d)\big]\;}
$$

(an average over training draws — the typical posterior an ideal observer reaches
after $N$ presentations).

### The number of inferred states grows with training

The two terms of $d$ pull in opposite directions as $N$ grows, and their balance
is the mathematical content of "it's the data, not a parameter."

**Prior (Occam), decreasing.** By Stirling,

$$
2\log\Gamma\!\Big(\tfrac N2\Big)-\log\Gamma(N)= -N\log 2-\tfrac12\log N+\text{const},
$$

so the CRP penalizes the extra cluster at a rate of $\log 2\approx0.69$ **per
observation**: a priori, splitting $N$ items into two prespecified equal groups
becomes exponentially less likely.

**Evidence, increasing.** With variance held fixed at $\sigma^2$, the one-state
model must explain data of spread $\sigma^2+\mu^2$ with a Gaussian of variance
$\sigma^2$ centred near $0$, paying $\mu^2/2\sigma^2$ in expected log-likelihood
per observation relative to the correctly split model. So the expected Bayes
factor grows linearly, $\mathbb E[\text{log BF}]\approx N\,\mu^2/(2\sigma^2)$.

**Net.** Combining,

$$
\mathbb E[d]\;\approx\;N\!\left(\frac{\mu^2}{2\sigma^2}-\log 2\right)+\log\alpha+O(\log N).
$$

Two states eventually win **iff the environments are distinct enough that evidence
beats Occam**, $\dfrac{\mu^2}{2\sigma^2}>\log 2$ (i.e. $\mu/\sigma>\sqrt{2\log2}\approx1.18$).
Then $P(2\text{ states}\mid N)$ climbs monotonically from $\approx0$ to $\approx1$
as training accumulates. With $\mu=1,\ \sigma=0.7$ the slope is
$1.02-0.69=0.33>0$, so the posterior rises from
$P(2)\approx0.32$ at $N=6$ (few observations — the partial-remapping regime,
cf. Leutgeb) to $P(2)\approx0.87$ at $N=52$ (many — the complete-remapping regime,
cf. Wills). (The exact
numbers are the Monte-Carlo average; since $\operatorname{logistic}$ is concave
for $d>0$, $\mathbb E[\operatorname{logistic}(d)]\le\operatorname{logistic}(\mathbb E d)$,
so the averaged curve sits below the plug-in estimate — the linear form captures
the trend, not the precise height.)

This is the crux: **nothing but $N$ changes**. No sharpness knob and no per-study
parameter — an ideal observer differentiates the two contexts *more* the more it
has been trained, purely because the evidence accumulates.

### From inferred states to differentiation

The link back to §B.1 is immediate. The observer's confidence that there are two
contexts, $P(2\text{ states}\mid N)$, *is* a differentiation signal: when it is
low (few training observations) the two context representations stay merged (small
$g$); when it is high (many observations) they are driven apart (large $g$). So the
differentiation $g$ is a monotone function of $P(2\text{ states})$, which — by
§B.4 — grows with training. **The model does not carry out this inference**; it
simply adopts the resulting monotone $g(\text{training})$ and lets the network
(§B.1–B.2) produce the remapping. In one line,

$$
\underbrace{\text{training }N}_{\text{data}}\;\longrightarrow\;\underbrace{P(2\text{ states})}_{\text{ideal observer, §B.3}}\;\longrightarrow\;\underbrace{g}_{\text{differentiation, §B.1}}\;\longrightarrow\;\text{Leutgeb (gradual) vs.\ Wills (abrupt).}
$$

| quantity | formula | role |
|---|---|---|
| cluster marginal likelihood | $\log p(\mathbf x)$ | evidence for one cluster |
| CRP prior log-ratio | $\log\alpha+2\log\Gamma(N/2)-\log\Gamma(N)$ | Occam penalty for splitting |
| posterior over states | $P(2\mid N)=\mathbb E[\operatorname{logistic}(d)]$ | grows with training |
| net log-odds slope | $\mu^2/2\sigma^2-\log 2$ | $>0$ ⇒ two states eventually win |

Only $g$ and the mechanism of §B.1–B.2 enter the model itself (`context_vector`).

---

# Antithetic sampling of theta sweeps (Figure 1)

Each theta cycle carries one stochastic rollout $\tilde z_k$ of the generator, and
the cycles are averaged into a Monte-Carlo estimate of the state distribution
(Eq. 5),

$$
\hat z(t)=\frac1K\sum_{k=1}^{K}\tilde z_k(t)\;\approx\;z(t).
$$

Vollan et al. (2025, *Nature* 639:995–1005) report that sweep direction alternates
stereotypically left/right across successive theta cycles, and argue this is an
efficient "look around" maximizing cumulative coverage. Alternation is exactly
what the classical **antithetic variates** construction produces. This part sets up
the sampler ($\S$C.1–C.3), defines the coupling ($\S$C.4), shows that the branch
flip is *exact* for a symmetric choice point but not otherwise ($\S$C.5), gives the
dwell-time correlation ($\S$C.6), derives a closed form for the variance reduction
that accounts for every number in Fig. 1b ($\S$C.7), states the condition under
which the sampled rollouts remain faithful to the master equation ($\S$C.8), and
exhibits a mechanism that satisfies it by construction ($\S$C.9). Symbols are local to
this part; $\tau$ here is the integration constant of Eq. 1, not a CA1 decay
constant.

## C.1 The generator and its escape rates

An environment is $n$ discrete states with a directed graph of allowed
transitions. Every allowed transition carries the *same* base rate $k$ — the only
free parameter of the environment — and the diagonal is fixed by conservation of
probability:

$$
W_{ij}=k\ \ \text{if } i\to j \text{ is allowed}\ (i\neq j),\qquad
W_{ij}=0\ \ \text{otherwise},\qquad
W_{ii}=-\!\!\sum_{j\neq i}\!W_{ij}.
$$

So the **escape rate** of a state — the rate that appears in the dwell-time draw —
is not chosen but derived, as the base rate times the out-degree $d_i$:

$$
\boxed{\;r_i=\frac{-W_{ii}}{\tau}=\frac{k\,d_i}{\tau}\;}
$$

For the T-maze of Figure 1 ($k=1$, $\tau=1$, $n=30$ states ordered
$[\text{stem }0..13,\ \text{left }14..21,\ \text{right }22..29]$, choice point
$c=13$) this takes exactly three values:

| states | $d_i$ | $r_i$ | mean dwell $1/r_i$ |
|---|---|---|---|
| stem $0$–$12$, arm interiors | $1$ | $k=1$ | $1$ unit $\approx 20.8$ ms |
| choice point $13$ | $2$ | $2k=2$ | $0.5$ units $\approx 10.4$ ms |
| arm ends $21,29$ | $0$ | $0$ (absorbing) | — |

With $T=6$ model units $=125$ ms, one unit is $20.8$ ms, so a sweep makes $\approx6$
jumps per cycle and reaches $\approx6$ states $\approx24$ cm ahead — the range in
Fig. 1B.

## C.2 The law of a rollout

The internal state obeys $z(t)=x\exp\!\big((t/\tau)W\big)$ (Eq. 3): the
distribution of a continuous-time Markov chain with generator $W/\tau$. A rollout
is one sample path of that chain, which from state $s$ dwells for
$\Delta t\sim\mathrm{Exponential}(r_s)$ and then jumps to $j$ with the embedded
jump-chain probability

$$
p(s\to j)=\frac{W_{sj}}{-W_{ss}}=\frac{W_{sj}}{\sum_{l\neq s}W_{sl}} .
$$

At the choice point $p(c\to 14)=p(c\to 22)=\tfrac12$. Everywhere else $d_i=1$, so
the jump is deterministic: **the arm choice is the only branch point in the entire
T-maze rollout.**

## C.3 Inverse-transform construction

To make the randomness *couplable* across rollouts, both draws are generated by
inverse transform from an explicit stream of i.i.d. uniforms
$u_0,u_1,u_2,\dots\sim\mathcal U[0,1)$, two per jump. The dwell time inverts the
exponential CDF $F(t)=1-e^{-r_st}$,

$$
\Delta t=F^{-1}(u)=-\frac{\ln(1-u)}{r_s},
$$

and the destination inverts the discrete CDF taken over the neighbours of $s$ in
**ascending state index** $\mathcal N(s)=(j_1<j_2<\dots<j_m)$,

$$
C_a=\sum_{b\le a}p(s\to j_b),\qquad
\text{destination}=j_{a^\*},\quad a^\*=\min\{a:\;C_a>u\}.
$$

The decoded sweep is this jump path read out at $n_{\mathrm{bins}}$ equally spaced
phases $t_b=bT/(n_{\mathrm{bins}}-1)$ as the state occupied at $t_b$
(right-continuous step interpolation).

## C.4 The antithetic pair

Draw one stream $u=(u_0,u_1,\dots)$ and drive the two members of a pair with

$$
\text{member 1}:\ (u_0,u_1,u_2,\dots),\qquad
\text{member 2}:\ (1-u_0,\,1-u_1,\,1-u_2,\dots).
$$

Since $U\sim\mathcal U[0,1)$ implies $1-U\sim\mathcal U[0,1)$, **each member is
marginally an exact draw** from the chain; only the *joint* law of the pair is
altered. Unbiasedness of $\hat z$ follows immediately and requires nothing else.

## C.5 The branch flip is exact only for a symmetric choice point

At the choice point $m=2$ with $p=(\tfrac12,\tfrac12)$, and the neighbours in
ascending index order are (left $=14$, right $=22$), so $C=(\tfrac12,1)$ and

$$
\text{destination}=\text{left}\iff u<\tfrac12 .
$$

Because $u<\tfrac12\iff 1-u\ge\tfrac12$, the two members take opposite arms **with
probability one**. Nothing in the code refers to "left" or "right": the flip is a
consequence of complementing the uniform that drives the branch.

This exactness is special to $p=\tfrac12$. For a binary branch with probabilities
$(p,1-p)$, member 1 takes branch 1 iff $u<p$ and member 2 iff $1-u<p$, i.e. iff
$u>1-p$. Exactly one of them takes branch 1 when $u<\min(p,1-p)$ or
$u>\max(p,1-p)$, so

$$
\boxed{\;\Pr[\text{opposite branches}]=2\min(p,1-p)\;}
$$

which equals $1$ **iff** $p=\tfrac12$ (e.g. $p=0.35$ alternates on only $70\%$ of
pairs, as verified numerically). Perfect alternation is therefore a prediction for
*symmetric* choice points; a biased branch should alternate less.

## C.6 The dwell times are anticorrelated too

$\Delta t(u)=-\ln(1-u)/r$ is increasing in $u$, so the partner's dwell
$-\ln(u)/r$ is *decreasing*: one member runs fast, the other slow. Both are
$\mathrm{Exponential}(r)$, so with $r=1$ (mean and variance $1$),

$$
\mathbb E\big[(-\ln U)(-\ln(1-U))\big]=\int_0^1\ln u\,\ln(1-u)\,du .
$$

Expanding $\ln(1-u)=-\sum_{m\ge1}u^m/m$ and using
$\int_0^1u^m\ln u\,du=-1/(m+1)^2$,

$$
\int_0^1\ln u\ln(1-u)\,du=\sum_{m\ge1}\frac{1}{m(m+1)^2}
=\sum_{m\ge1}\left[\frac1m-\frac1{m+1}-\frac1{(m+1)^2}\right]
=1-\big(\zeta(2)-1\big)=2-\frac{\pi^2}{6},
$$

the two sums being a telescoping series and $\zeta(2)-1$. Hence
$\mathrm{Cov}=2-\pi^2/6-1$ and, since both variances are $1$,

$$
\boxed{\;\mathrm{Corr}\big(\Delta t_1,\Delta t_2\big)=1-\frac{\pi^2}{6}\approx-0.645\;}
$$

(confirmed numerically at $-0.6451$). This part of the coupling acts even on a 1-D
track where there is no branch, and it is what sharpens the look-ahead distance.

## C.7 How much variance is removed

Fix a read-out time $t$ and a state $i$, and let $X_k=\mathbb 1[\tilde s_k(t)=i]$,
so $\hat z_i=\frac1K\sum_kX_k$ with $\mathbb E[X_k]=z_i$ and
$\sigma_i^2=\operatorname{Var}[X_k]=z_i(1-z_i)$. For independent sweeps
$\operatorname{Var}[\hat z_i]=\sigma_i^2/K$. For $K/2$ *independent pairs* with
within-pair correlation $\rho_i=\mathrm{Corr}(X_1,X_2)$,
$\operatorname{Var}(X_1+X_2)=2\sigma_i^2(1+\rho_i)$, and therefore

$$
\operatorname{Var}_{\text{anti}}[\hat z_i]
=\frac{1}{K^2}\cdot\frac K2\cdot 2\sigma_i^2(1+\rho_i)
=\frac{\sigma_i^2(1+\rho_i)}{K}.
$$

Throughout, "variance reduction" means the **ratio** of the two variances, not
their difference — a dimensionless factor, equal to 1 when the coupling buys
nothing:

$$
\boxed{\;\text{variance reduction factor}\;\equiv\;
\frac{\operatorname{Var}_{\text{indep}}[\hat z_i]}{\operatorname{Var}_{\text{anti}}[\hat z_i]}
=\frac{1}{1+\rho_i}\;}
$$

The ratio is the useful summary because both variances fall as $1/K$, so it is
(to leading order) independent of $K$ and comparable across estimators with quite
different scales; their *difference* would inherit both the $1/K$ and the units of
whatever is being estimated.

Both schemes fall as $1/K$ (Fig. 1b B); the coupling changes only the prefactor.
Two consequences:

- **The branch indicator is estimated exactly.** Exactly one member of each pair
  enters the left arm, so $X_1+X_2=1$ identically,
  $\operatorname{Var}(X_1+X_2)=0$, hence $\rho=-1$ and
  $\operatorname{Var}[\hat P(\text{left})]=0$ at every $K$ — the "exact" row of
  Fig. 1b C. This is the $\rho=-1$ floor of the formula above.
- **Everything else depends on how negative $\rho_i$ is.** Inverting
  $1/(1+\rho_i)$ on the measured reductions:

| estimator | measured reduction | implied $\rho$ |
|---|---|---|
| $P(\text{left})$ | exact | $-1$ (exactly) |
| arm occupancy | $\approx47\times$ | $\approx-0.98$ |
| look-ahead | $\approx1.91\times$ | $\approx-0.48$ |
| value (goal in one arm) | $\approx1.40\times$ | $\approx-0.29$ |
| belief $z(t)$ | $\approx1.16\times$ | $\approx-0.14$ |
| left$-$right value contrast | $\approx1.05\times$ | $\approx-0.05$ |

The fine-grained belief gains little because the two members occupy *disjoint*
arms: their state indicators are supported on different states and so cannot
cancel, leaving only the weaker dwell-time coupling of $\S$C.6. The antisymmetric
value contrast gains nothing because for a quantity that flips sign across the
branch, the negative coupling works *against* the branch balancing — the classical
statement that antithetic variates help for functions monotone in the driving
uniforms and can fail for functions that are odd about $u=\tfrac12$.

## C.8 Consistency of the microscopic sampling with the macroscopic dynamics

The master equation is a *deterministic macroscopic* law, $z(t)=x\exp((t/\tau)W)$,
while the model realizes *single stochastic rollouts* and averages them (Eq. 5).
Coupling those rollouts antithetically changes how they are drawn, so when is the
microscopic average still faithful to the macroscopic dynamics? The requirement
is

$$
\mathbb E\big[\hat z(t)\big]=z(t),\qquad
\hat z(t)=\frac1K\sum_{k=1}^{K}\tilde z_k(t).
$$

**The condition is on the marginals alone.** By linearity of expectation,

$$
\mathbb E\big[\hat z(t)\big]=\frac1K\sum_{k=1}^{K}\mathbb E\big[\tilde z_k(t)\big],
$$

which equals $z(t)$ **if and only if each $\tilde z_k$ is marginally distributed as
the CTMC**. Independence is never used. In probabilistic language, the $K$
rollouts are a **coupling** of $K$ copies of the master-equation law — any joint
distribution with the prescribed marginals — and every coupling has the same mean;
couplings differ only in variance. Hence the design rule:

> **Couple the samples, never the marginals.**

The guarantee is stronger than it looks: each $\tilde z_k$ is a whole sample
*path* with the correct law, so *all* finite-dimensional distributions agree, not
just the occupancy at one $t$. Every path functional is therefore unbiased too —
the discounted return $\int_0^T e^{-\lambda t}r(\tilde s(t))\,dt$ included.

**Scheme 1: antithetic streams ($\S$C.4).** If $U\sim\mathcal U[0,1)$ then
$1-U\sim\mathcal U[0,1)$, so the complemented stream is again i.i.d. uniform. A
rollout is a deterministic function of its stream (inverse transform, $\S$C.3),
so the partner is an exact CTMC draw and $\hat z$ is unbiased. Note this holds for
*any* branch probabilities: the $2\min(p,1-p)$ of $\S$C.5 governs how often the
pair splits, i.e. the variance benefit, and has no bearing on the mean.

**Scheme 2: locking the pair order (`order_key`).** Here the two members are
assigned to cycles by a rule that depends on their *outcome* (which arm each
took). That is deliberately **not** a coupling of the per-cycle marginals:
conditioned on being cycle $n$, the sweep is no longer CTMC-distributed (with the
left-first rule, even cycles are always left). But $\hat z$ depends only on the
unordered multiset $\{\tilde z_1,\tilde z_2\}$ of each pair, which the rule leaves
untouched, so the aggregate identity survives for even $K$. The distinction is
real and measurable — per-cycle $|z|\approx30$ against $\approx2$–$3$ for the
honest schemes. **Consequence:** never read a single cycle, or an odd-length
window, from the locked-order scheme; it is valid only as a $K$-cycle average.

**How to ensure it in practice.** `test_micro_macro_consistency.py` checks
$\mathbb E[\text{microscopic}]=\text{macroscopic}$ against $\exp((t/\tau)W)$ for
every scheme on five environments, at both levels (aggregate and per-cycle), and
asserts the expected per-cycle failure of the locked order above so it cannot
silently regress. Two
statistical points make the test trustworthy rather than decorative: cells are
tested only where the expected count reaches 25 (the model has states with
$z\sim10^{-18}$, where a normal approximation would manufacture large $|z|$ from a
single stray count), and the per-cycle test uses the **exact null standard error**
$\sqrt{z_i(1-z_i)/R}$ — under a consistent scheme a single sweep's indicator is
Bernoulli$(z_i(t))$ whatever the coupling — rather than the empirical sd, which is
biased low at small counts and produces spurious failures. Add a scheme to the
list and the check comes for free.

## C.9 A mechanism that provably samples the generator *and* is antithetic

$\S$C.8 says a coupling is safe iff it leaves every marginal intact. That is a
*condition*, not a construction: an adaptation rule that suppresses whatever was
just sampled generically **does** disturb the marginal, and then the microscopic
average drifts away from the master equation. This section gives a mechanism for
which marginal correctness is structural — provable, with no symmetry assumption
and no parameter to calibrate — whose antithetic structure comes precisely from
an adaptation that discourages repeating a path.

### The invariance that does the work

A rollout is a *deterministic* function $\Phi$ of its uniform stream (inverse
transform, $\S$C.3). So if $\tau:[0,1)\to[0,1)$ preserves Lebesgue measure and is
applied coordinatewise to the stream $u=(u_0,u_1,\dots)$, then $\tau(u)$ is again
i.i.d. uniform and

$$
\boxed{\;\Phi(\tau(u))\ \stackrel{d}{=}\ \Phi(u)\;}
$$

**Any coordinatewise measure-preserving reparameterization of the stream leaves
each rollout marginally an exact draw of the generator**, no matter what
dependence it induces *across* rollouts. Three instances:

| $\tau$ | scheme |
|---|---|
| $u\mapsto 1-u$ (reflection) | antithetic pairs ($\S$C.4) |
| $u\mapsto u+c \bmod 1$ (rotation) | lattice sampling; $c=\tfrac12$ flips a symmetric branch |
| place $u$ inside a chosen stratum | stratification — used below |

### Ingredient 1: stratify the branch coordinate

The branch is drawn by inverse-CDF from the strata
$I_j=[C_{j-1},C_j)$, whose lengths are exactly the branch probabilities
$|I_j|=p_j$ ($\S$C.3). Suppose we first pick a stratum index $j$ with probability
$p_j$ and then place the uniform *inside* it, $u\sim\mathcal U(I_j)$. Then for any
Borel $A\subseteq[0,1)$,

$$
\Pr[u\in A]=\sum_j p_j\,\frac{|A\cap I_j|}{|I_j|}=\sum_j |A\cap I_j|=|A| ,
$$

so $u$ is marginally uniform and the rollout is marginally exact. This holds **for
any branch law $p$** — no symmetry is required. It replaces the mirror reflection
used earlier, which needed the two arms to be isomorphic sub-chains, and it is the
step that removes the need to calibrate anything.

### Ingredient 2: choose the stratum by depletion, not by a free gain

Give each branch a **quota** $m_j=p_jM$ over a block of $M$ theta cycles, and let
the adaptation state be the *residual* quota. On cycle $n$ of the block,

$$
\boxed{\;\Pr\big[X_n=j\mid \mathcal F_{n-1}\big]
=\frac{m_j-c^{(n-1)}_j}{M-(n-1)}\;}
$$

where $c^{(n-1)}_j$ counts how often branch $j$ has already been used in this
block. Using a branch decrements its own availability — a use-dependent
depression — and the block refills it. This is sampling **without replacement**
from a reservoir holding $m_j$ tokens of type $j$.

**Provable marginal correctness.** The sequence $X_1,\dots,X_M$ so generated is a
uniformly random permutation of that multiset, hence *exchangeable*, so

$$
\Pr[X_n=j]=\frac{m_j}{M}=p_j \qquad\text{for every } n .
$$

Together with Ingredient 1 this makes every cycle marginally an exact rollout of
the generator. Checked numerically (per-cycle $\max|z|$, no calibration used):
$2.6$ at $p=\tfrac12$ with $M=2$, $2.4$ at $p=0.75$ and $3.4$ at $p=0.25$ with
$M=4$. Contrast the locked antithetic order of $\S$C.8, which is only
*aggregate* unbiased ($\max|z|=30$ per cycle).

**Provable antithetic structure.** Sampling without replacement is negatively
associated. For the branch frequency over the first $k$ cycles of a block the
finite-population correction gives exactly

$$
\operatorname{Var}\big[\hat p_k\big]=\frac{p(1-p)}{k}\cdot\frac{M-k}{M-1}
\;<\;\frac{p(1-p)}{k}\quad (k\ge2),
$$

reaching **exactly zero at $k=M$**, where the block is balanced by construction.
Measured against this formula at $M=8$, $p=\tfrac12$:

| $k$ | measured | theory | independent |
|---|---|---|---|
| $1$ | $0.2500$ | $0.2500$ | $0.2500$ |
| $2$ | $0.1062$ | $0.1071$ | $0.1250$ |
| $4$ | $0.0360$ | $0.0357$ | $0.0625$ |
| $8$ | $0.0000$ | $0.0000$ | $0.0313$ |

Note $k=1$ shows *no* gain, and must not: the per-cycle marginal is exactly right,
so the entire benefit lives in the dependence between cycles — which is the
content of $\S$C.8 read in reverse. At the level of whole sweeps the arm-occupancy
estimate is $58\times$ less variable than independent sampling ($M=2$, $K=8$),
comparable to the $\approx47\times$ of the reflection-based pairs ($\S$C.7).
$M=2$ *is* antithetic pairing; larger $M$ spreads the balancing over a longer
window, giving a less rigid but still exactly balanced sequence.

**What it leaves on the table.** Depletion couples only the branch coordinate;
the dwell times are redrawn independently, so this scheme does not inherit the
$\approx1.9\times$ look-ahead sharpening that $u\mapsto1-u$ gets from its
anticorrelated dwells ($\S$C.6). Applying a measure-preserving $\tau$ to the
remaining coordinates as well recovers it, and is equally licensed by the boxed
invariance above.

### Circuit reading, and the one thing the math insists on

The mechanism is a winner-take-all over a **depleting resource**: each branch
carries an availability that is decremented by use and replenished on the block
timescale, and selection is proportional to what remains. Biophysically this is
short-term synaptic depression, or an adaptation current, gating a competitive
selection stage — the same ingredients as a half-centre oscillator.

The one non-negotiable is the *normalization*. The selection probability must be
the residual quota divided by the number of cycles **remaining**,
$(m_j-c_j)/(M-n+1)$, which is a *divisive* (shunting) competition over the
available pool. An unnormalized subtractive adaptation with a free gain — drive
minus $g\,a_j$ — does not compute this, and has no reason to reproduce $p_j$;
that is exactly why such a circuit has to be calibrated to the generator or
reweighted, whereas the depleting-reservoir rule is correct by construction.
Adaptation *per se* does not bias the sampler; adaptation with an arbitrary gain
does.

## C.10 Neural implementation

$\S$C.9 is an algorithm; this section asks what circuit could run it. Three things
have to be computed: the branch law $p_j$, a state variable that discourages
re-sampling the same branch, and a selection stage.

### The branch law is already in the synapses

From $\S$C.2,

$$
p_j=\frac{W_{cj}}{-W_{cc}}=\frac{W_{cj}}{\sum_{k\neq c}W_{ck}} ,
$$

the outgoing CA3 weight from the choice-point assembly **divisively normalized by
the total outgoing weight**. Divisive normalization across a population is one of
the best-established cortical operations, so the generator's branch law needs no
extra machinery: it *is* the normalized synaptic drive out of the current
assembly. Nothing below has to represent $p_j$ separately.

### Implementation A: a depleting synaptic resource

Because the quotas are exhausted exactly when the block ends,
$\sum_j\big(m_j-c_j\big)=M-(n-1)$, the rule of $\S$C.9 can be rewritten with no
reference to $n$ at all:

$$
\boxed{\;\Pr[X_n=j\mid\mathcal F_{n-1}]=\frac{R_j}{\sum_k R_k},
\qquad R_j=m_j-c_j\;}
$$

*Selection proportional to the remaining resource, divisively normalized by the
total remaining.* That is a circuit, and a conventional one:

| quantity | neural substrate |
|---|---|
| $R_j$ | available presynaptic resource at the choice-point $\to$ arm-$j$ synapses (short-term synaptic depression) |
| $\sum_k R_k$ | the same divisive-normalization pool as above — a shared interneuron population reading total available drive |
| the draw | stochastic winner-take-all over the normalized drive |

Adaptation here is ordinary use-dependent synaptic depression: sweeping an arm
consumes that pathway's resource, which is exactly "reduce the probability of
repeatedly sampling the same path".

Two conditions are needed for *exactness*. (i) Depletion must be **quantal** —
one unit per use — which a small vesicle pool supplies naturally. (ii) Resource
must be **replenished to quota at block boundaries**, i.e. recovery locked to a
rhythm $M$ theta cycles long, rather than a free exponential. This second point is
a genuine constraint and worth stating plainly: ordinary exponential recovery does
*not* implement sampling without replacement. It preserves the qualitative
antithetic structure — use suppresses repeats, successive cycles are negatively
dependent — but the marginal is then only *approximately* $p_j$, and would have to
be verified ($\S$C.8) or calibrated. The lesson of the half-centre motif reappears
here, localized to the recovery law.

### Implementation B: a slow ramp read against the weights

Depletion is not the only measure-preserving option. Using the *rotation* instance
of the invariance theorem instead, let a phase advance by $1/M$ per theta cycle
and wrap, with a fresh random offset $V\sim\mathcal U[0,1)$ once per block:

$$
\varphi_n=\big(V+n/M\big)\bmod 1,
\qquad
\text{select branch } j \iff \varphi_n\in I_j,\ \ |I_j|=p_j .
$$

The selection is a **comparator between a ramping phase and the normalized
synaptic thresholds**. Its properties:

- *Provably exact per cycle.* $V$ uniform $\Rightarrow\varphi_n$ uniform
  $\Rightarrow\Pr[X_n=j]=|I_j|=p_j$ for every $n$. Verified: per-cycle
  $\max|z|=2.2,\,2.9,\,2.8,\,3.3$ at $p=\tfrac12\,(M{=}2)$, $0.75$, $0.25$
  $(M{=}4)$ and $\tfrac12\,(M{=}8)$.
- *Exactly balanced.* The $M$ phases are equally spaced, so every block contains
  exactly $p_jM$ selections of $j$ — verified as a *constant* per-block count, not
  a distribution. (The urn of $\S$C.9 also hits the quota, but in random order;
  the ramp fixes the order too.)
- *Cheap.* One random number per block instead of one per cycle, and no
  bookkeeping: the "adaptation" is the monotone ramp itself, which cannot revisit
  a stratum until the block completes.

Neurally this is a theta-nested slower oscillation, or a ramping adaptation
current that resets — both familiar in hippocampus — feeding a threshold
comparison against the recurrent weights. A real oscillator's phase also
*diffuses*, which turns out to matter: $\S$C.11 treats that case, and it is both
the physically correct version and the one that reproduces the observed
alternation rate.

### Why the earlier motif was neither

A subtractive adaptation with a free gain, $\text{drive}-g\,a_j$, is neither the
residual-quota *ratio* of A nor the measure-preserving *rotation* of B, so it has
no reason to reproduce $p_j$ and in general does not. Restated: **adaptation does
not bias the sampler; adaptation with an arbitrary gain does.** What matters is
that the competition be divisive over the depleting pool, or that the state
variable act on the uniform stream by a measure-preserving map.

### Beyond the branch

Only the branch coordinate has been stratified. Stratifying the dwell times too
means a lattice in several coordinates,
$u^{(i)}_n=\big(V_i+n\,a_i/M\big)\bmod1$ with incommensurate increments $a_i$ —
several oscillators at different frequencies — which would additionally recover
the dwell-time anticorrelation of $\S$C.6 and so the look-ahead sharpening that
branch-only depletion forgoes.

### Two circuits, two predictions

A and B are behaviourally distinguishable, which makes this more than a
plausibility argument:

- **A is use-dependent.** Externally forcing the same arm on two successive
  cycles should *suppress* that arm afterwards, because its resource is depleted.
- **B is autonomous.** The phase advances regardless of what was sampled, so an
  imposed repeat should leave the subsequent sequence unchanged; instead the
  upcoming direction should be predictable from the phase of a slow theta-nested
  rhythm.
- **Both** predict that at a *biased* choice point the sweep frequencies track the
  branch statistics $p_j$ rather than splitting 50/50 — the experiment that
  separates either of them from a fixed alternator.

## C.11 The theta-skipping pacemaker with phase diffusion

$\S$C.10 Implementation B selected the branch with a ramping phase. This section
writes that mechanism down in its correct form — with a *diffusing* phase — and
derives, in closed form, its alternation rate and the variance of the Eq.-5
estimator. Everything below is verified numerically to 3–4 decimals.

### The model

Let $\varphi_k\in[0,1)$ be the pacemaker phase on theta cycle $k$:

$$
\boxed{\;\varphi_k=\big(\varphi_{k-1}+\tfrac12+\xi_k\big)\bmod 1,
\qquad \varphi_0\sim\mathcal U[0,1),\quad \xi_k\stackrel{\text{iid}}{\sim}\mathcal N(0,\sigma^2)\;}
$$

and select branch $j$ on cycle $k$ iff $\varphi_k\in I_j$, where the $I_j$ are the
inverse-CDF strata of $\S$C.3, $|I_j|=p_j$. Unrolling,

$$
\varphi_k=\Big(\varphi_0+\frac k2+W_k\Big)\bmod 1,
\qquad W_k=\sum_{i\le k}\xi_i\sim\mathcal N(0,k\sigma^2).
$$

**The noise must accumulate.** It is $W_k$, with variance growing like $k\sigma^2$,
that enters — not an independent $\varepsilon_k$ of fixed variance added to a
frozen $\varphi_0$. The two differ in exactly one respect, ergodicity, and
$\S$C.11.5 shows that this is what decides whether the estimator converges.

### C.11.1 Marginal consistency, for every $\sigma$ and every $p$

**Lemma.** If $\varphi\sim\mathcal U[0,1)$ and $\eta$ is *any* random variable
independent of $\varphi$, then $(\varphi+\eta)\bmod 1\sim\mathcal U[0,1)$.

*Proof.* Work on the circle $\mathbb T=\mathbb R/\mathbb Z$ and compare Fourier
coefficients. For $n\neq0$,
$\mathbb E\big[e^{2\pi i n(\varphi+\eta)}\big]
=\mathbb E\big[e^{2\pi i n\varphi}\big]\,\mathbb E\big[e^{2\pi i n\eta}\big]=0$,
since $\mathbb E[e^{2\pi in\varphi}]=0$ for the uniform. All non-constant Fourier
coefficients vanish, so the law is uniform. $\square$

Applying it with $\eta=k/2+W_k$ gives $\varphi_k\sim\mathcal U[0,1)$ and hence

$$
\Pr[X^k=j]=\Pr[\varphi_k\in I_j]=|I_j|=p_j \qquad\text{for every }k .
$$

Two things are worth noting. First, this is *stronger* than an argument from the
drift alone: shifting by the constant $\tfrac12$ is measure preserving, but so is
convolving with the noise, and it is the lemma — not the constant shift — that
covers $W_k$. Second, unbiasedness comes entirely from $\varphi_0$ being uniform
and is completely insensitive to $\sigma$. **The noise is therefore free with
respect to marginal consistency**, and can be chosen purely to fit the
alternation rate.

### C.11.2 Autocorrelation of the branch indicator

Take the symmetric case $p=\tfrac12$, $I=[0,\tfrac12)$, and let
$X_k=\mathbb 1[\varphi_k\in I]$. Expanding $\mathbb 1_I$ in Fourier series gives
coefficients $c_0=\tfrac12$ and $c_n=1/(\pi i n)$ for odd $n$ (even $n\neq0$
vanish), so with $\varphi_{k+\ell}=\varphi_k+\ell/2+W_\ell$,

$$
\boxed{\;\rho_\ell=(-1)^{\ell}\,\frac{8}{\pi^{2}}
\sum_{n\ \mathrm{odd}\ \ge1}\frac{e^{-2\pi^{2}n^{2}\ell\sigma^{2}}}{n^{2}}\;}
$$

The sign alternates — negative at odd lags, which *is* the antithetic property —
and the magnitude decays exponentially at rate $2\pi^2\sigma^2$ per cycle. At
$\sigma=0$ the sum is $\pi^2/8$ and $\rho_\ell=(-1)^\ell$ exactly: a perfect
period-2 sequence. Measured against the formula at $\sigma=0.07$:

| lag $\ell$ | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| measured | $-0.777$ | $+0.684$ | $-0.614$ | $+0.554$ | $+0.455$ |
| exact | $-0.777$ | $+0.684$ | $-0.613$ | $+0.553$ | $+0.454$ |

### C.11.3 Alternation rate

The probability that consecutive cycles switch side follows from the same
expansion. Writing $\Lambda$ for the triangle wave
$\Lambda(x)=2\min(\{x\},1-\{x\})$, a shift by $d$ flips the branch with
probability $\Lambda(d)$, so $q=\mathbb E_\xi[\Lambda(\tfrac12+\xi)]$ and

$$
\boxed{\;q(\sigma)=\frac12+\frac{4}{\pi^{2}}
\sum_{n\ \mathrm{odd}\ \ge1}\frac{e^{-2\pi^{2}n^{2}\sigma^{2}}}{n^{2}}\;}
$$

with $q(0)=1$ (deterministic alternation) and $q(\infty)=\tfrac12$ (chance). For
small $\sigma$ only $|\xi|$ matters, $\Lambda(\tfrac12+\xi)=1-2|\xi|$, giving the
useful linear law

$$
q(\sigma)\;\approx\;1-2\sigma\sqrt{2/\pi}\;=\;1-1.596\,\sigma .
$$

| $\sigma$ | 0.02 | 0.04 | 0.07 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| measured | 0.9684 | 0.9359 | 0.8886 | 0.8408 | 0.6841 |
| exact | 0.9681 | 0.9362 | 0.8883 | 0.8404 | 0.6841 |
| $1-1.596\sigma$ | 0.9681 | 0.9362 | 0.8883 | 0.8404 | 0.6808 |

### C.11.4 Variance of the Eq.-5 estimator

For a stationary sequence $K\operatorname{Var}[\hat p_K]\to\sigma_X^2(1+2\sum_{\ell\ge1}\rho_\ell)$.
Summing the geometric series in $\S$C.11.2 term by term ($\sigma_X^2=\tfrac14$):

$$
\boxed{\;K\operatorname{Var}\big[\hat p_K\big]\;\longrightarrow\;
\frac14\left[1-\frac{16}{\pi^{2}}\sum_{n\ \mathrm{odd}\ \ge1}
\frac{1}{n^{2}\big(1+e^{2\pi^{2}n^{2}\sigma^{2}}\big)}\right]\;}
$$

As $\sigma\to0$ each term tends to $1/(2n^2)$, the bracket tends to
$1-\tfrac{16}{\pi^2}\cdot\tfrac{\pi^2}{16}=0$, and the branch frequency is
estimated *exactly* — recovering the deterministic pacemaker. As
$\sigma\to\infty$ the bracket tends to $1$ and the estimator degrades to
independent sampling, $\operatorname{Var}=p(1-p)/K$. In between:

| $\sigma$ | measured ($K=2048$) | exact | reduction vs independent |
|---|---|---|---|
| 0.04 | 0.0241 | 0.0243 | $10.3\times$ |
| 0.07 | 0.0422 | 0.0425 | $5.9\times$ |
| 0.15 | 0.0906 | 0.0908 | $2.8\times$ |

So $\sigma$ trades alternation fidelity against variance reduction along a single
curve, and both are known in closed form.

### C.11.5 Why the noise has to accumulate

Replace $W_k$ by an independent $\varepsilon_k\sim\mathcal N(0,\sigma^2)$, so that
$\varphi_k=\varphi_0+k/2+\varepsilon_k$. Marginal consistency still holds (the
lemma does not care). But the chain never forgets $\varphi_0$: conditional on it
the $X_k$ are independent, so by the law of total variance

$$
\operatorname{Var}\big[\hat p_K\big]\;\ge\;
\operatorname{Var}_{\varphi_0}\!\big[g(\varphi_0)\big],
\qquad
g(\varphi)=\tfrac12\sum_{a\in\{0,1/2\}}\Pr\big[(\varphi+a+\varepsilon)\bmod 1\in I\big],
$$

a floor independent of $K$. At $p=\tfrac12$ the two parities are complementary,
$g\equiv\tfrac12$, and the floor vanishes — which is why i.i.d. jitter looks fine
at a symmetric choice point. For $p\neq\tfrac12$ it is strictly positive: at
$p=0.75,\ \sigma=0.1$ the formula gives $0.01044$ against a measured plateau of
$0.01048$ at $K=2048$, where independent sampling would give $9\times10^{-5}$.

With accumulating noise $\varphi_k$ is an ergodic random walk on the circle, mixing
to uniform at rate $2\pi^2\sigma^2$; $\varphi_0$ is forgotten, the floor
disappears, and $\operatorname{Var}[\hat p_K]$ falls as $1/K$ (measured
$0.0204\to8.7\times10^{-5}$ over $K=8\ldots2048$ at $p=0.75$, slightly *below*
independent throughout). **This is the whole content of the distinction: i.i.d.
jitter is not diffusion, and only diffusion restores ergodicity.**

### C.11.6 Calibration, and multi-way branches

Matching the $79.8\%$ triplet alternation of Vollan et al. gives

$$
\sigma_u\approx0.075
\qquad(\text{triplet }0.797,\ \ q=0.881,\ \ \text{variance reduction}\approx5.5\times).
$$

Finally, the noise repairs the one structural failure of the deterministic
pacemaker. With $\sigma=0$ the phase takes only the two values
$\varphi_0,\varphi_0+\tfrac12$, so at a three-way choice point exactly two of the
three branches are ever visited — every session misses one entirely. Any
$\sigma\gtrsim0.07$ restores full coverage (all three branches visited in $100\%$
of sessions), because the phase then explores the whole circle. A single phase
still stratifies only one branch variable; independent decisions need
independent phases, as in the lattice of $\S$C.10.

| object | formula | code |
|---|---|---|
| escape rate | $r_i=-W_{ii}/\tau=k\,d_i/\tau$ ($\S$C.1) | `sample_trajectory` |
| dwell / destination | $-\ln(1-u)/r_s$; ascending-index CDF ($\S$C.3) | `sample_trajectory(u=…)` |
| antithetic streams | $u$ and $1-u$ ($\S$C.4) | `AntitheticStream` |
| flip probability | $2\min(p,1-p)$ ($\S$C.5) | — (prediction) |
| variance reduction factor | $\operatorname{Var}_{\text{indep}}/\operatorname{Var}_{\text{anti}}=1/(1+\rho_i)$ ($\S$C.7) | `analysis_antithetic_variance.py` |
| micro/macro consistency | each rollout marginally CTMC $\Rightarrow\mathbb E[\hat z]=z$ ($\S$C.8) | `test_micro_macro_consistency.py` |
| marginal invariance | $\Phi(\tau(u))\stackrel{d}{=}\Phi(u)$ for measure-preserving $\tau$ ($\S$C.9) | — |
| quota depletion | $\Pr[X_n{=}j\mid\mathcal F]=(m_j-c_j)/(M-n+1)$ ($\S$C.9) | `test_micro_macro_consistency.py` |
| depleting-resource circuit | $\Pr[X_n{=}j]=R_j/\sum_k R_k$ — short-term depression + divisive normalization ($\S$C.10) | `test_micro_macro_consistency.py` |
| ramp circuit | $\varphi_n=(V+n/M)\bmod 1$ vs weight thresholds ($\S$C.10) | `test_micro_macro_consistency.py` |
| pacemaker + diffusion | $\varphi_k=(\varphi_{k-1}+\tfrac12+\xi_k)\bmod1$ ($\S$C.11) | `check_pacemaker.py` |
| alternation rate | $q(\sigma)=\tfrac12+\tfrac{4}{\pi^2}\sum_{n\,\mathrm{odd}}n^{-2}e^{-2\pi^2n^2\sigma^2}$ ($\S$C.11.3) | `check_pacemaker.py` |
