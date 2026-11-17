# Simulation details

Environment structure, policies, parameters, and analysis for each simulation.
Parameter values are the defaults in the `run(...)` functions of the
correspondingly numbered script in [`sims/`](sims/); every figure (PDF) is
reproducible with `python run_all.py`. Equation numbers refer to the note.

## Global conventions

- **Spatial scale.** Space is discretized into bins ("states"), each **`BIN_CM =
  4 cm`** (`hippo.BIN_CM`), applied consistently everywhere, so results are in
  physical units: a 40-bin linear track is 160 cm, a 52×44 arena is 208×176 cm
  (≈ the Pfeiffer–Foster arena), a 72-bin ring is a 288-cm circumference. All
  spatial axes are labelled in cm.
- **Time / theta.** The transition *rates* set the model time unit; one theta
  cycle is **`THETA_PERIOD = 6`** model units, identified with **125 ms**. The
  leak rate `λ` is the continuous-time discount: larger `λ` ⇒ steeper
  discounting ⇒ smaller / shorter-range fields.
- **Truncated resolvent (125 ms theta sweeps).** Every behavioural / predictive-
  map SR readout uses the **finite-depth resolvent** (Eq. 14),
  `R_T = (λI−W)⁻¹(I − e^{−(λI−W)T}) = ∫₀ᵀ e^{−λt} e^{Wt} dt` with `T =
  THETA_PERIOD` — the successor representation accumulated over a single theta
  sweep rather than the infinite horizon (`model.resolvent_truncated`,
  `output_ctx(..., T=THETA_PERIOD)`). This keeps local structure and suppresses
  the slow/global modes a full sweep would reach. The one exception is the
  **intrinsic tuning width** in Sim 3 (A–C), which is the *asymptotic* steady-
  state scale (width = `c/√λ`) and so uses the full resolvent.
- **Generator `W`.** A continuous-time Markov-chain generator: off-diagonal
  entries are non-negative transition rates, each row sums to zero
  (`project_to_generator`). The behavioral policy is baked into `W`.
- **Trained agent / optimal policy.** Unless a simulation explicitly studies
  learning (only Sim 2A, backward skew, does), we assume a **fully-trained agent
  executing the optimal policy** — `W` encodes that policy directly. Population
  read-outs formed by summing over start states (e.g. the reward-clustering
  density, `R.sum(axis=0)`) correspond to **randomized/uniform start positions**,
  not a single fixed start.
- **Internal state.** `z` is a row vector; `z(t) = x·exp((t/τ)W)` (Eq. 3),
  τ = 1 throughout.
- **Resolvent / SR.** `R = (λI−W)⁻¹` (Eq. 8); `R[i,j]` is the discounted future
  occupancy of `j` from `i`. The **place field of cell `j` is column `j`** of `R`
  (or of the truncated `R_T` for the behavioural sims).
- **DG context gating.** `ContextHippocampalModel`: effective generator
  `J_ij = W_ij b_j` with gate `b = cG` (Eqs. 17–18), rows re-balanced to sum to
  zero. `c` is the DG context vector, `G` the DG→CA3 gating weights.
- **Learning.** Online three-factor rule `ΔW_ij ∝ ε_j b_j z_i`, error
  `ε_j = ż_j − Σ_i z_i W_ij b_j` (Eqs. 20–21).
- **Output.** Figures are saved as PDF; panel labels are bold at the top-left.
  Stochastic simulations use seeded `numpy` generators (seeds below).

The six figures are grouped by theme: prediction/imagination dynamics (1);
predictive place-field structure and inference (2 — backward skew, reward
clustering, inferential reasoning); the multi-scale predictive map (3); and
context/memory/trajectory coding (4–6).

## Default parameters

Shared parameters are held at these defaults across all simulations; the last
column lists every deviation and why it is unavoidable. Environment geometry
(track/ring/arena size) and random seeds are listed per-simulation below and are
not repeated here.

| parameter | symbol | default | deviations (reason) |
|---|---|---|---|
| leak rate | `λ` | **0.20** | Sim 3 uses the CA1 leak-rate *distribution* `λ = 1/τ` (the multi-scale sim is *about* a spread of `λ`) |
| theta period | `T` | **6** (125 ms) | — (global `THETA_PERIOD`) |
| bin size | — | **4 cm** | — (global `BIN_CM`) |
| integration constant | `τ_int` | **1.0** | — (the `z(t)=x·exp((t/τ)W)` constant; distinct from the CA1 decay `τ=1/λ`) |
| generator base rate | `k` | **1.0** | — (per-step transition rate for track / T-maze / arena) |
| EC tuning width | `σ` | **1.6 bins** | — (Sims 1, 2, 4, 5; Sims 3 and 6 read fields directly, no EC bump) |
| DG gate ON fraction | `frac` | **0.6** | — (Sims 4, 5, 6) |
| DG gate OFF level | `lo` | **0.05** | — (Sims 4, 5, 6) |

Genuinely simulation-specific parameters (no shared default): Sim 2 backward-skew
learning rate `η = 0.08` (the only online-learning component); Sim 2 reward
policy `drift = 1.3`, `diffusion = 0.10`, `dwell_factor = 0.40`, `dwell_w = 6`
(goal-directed swimming); Sim 2 inference `n = 12` audio→visual→outcome triples;
Sim 5 context differentiation `DIFF_LO = 1.0` (matching Leutgeb), `DIFF_HI = 6.0`
(matching Wills); Sim 6 trajectory-gain `amp = 0.8`, `cutoff = 0.12`; Sim 3 width
scale `c ≈ 3.28 cm·s^½` (calibrated to the empirical median).

---

## Simulation 1 — Forward theta sweeps and phase precession (`sim1_theta_phase.py`)

**Experiments.** Johnson & Redish (2007); Vollan et al. (2025); O'Keefe & Recce
(1993); Skaggs et al. (1996). One mechanism: each theta cycle is a forward
rollout (Eq. 5).

All sweeps are decoded over one theta cycle, `T = THETA_PERIOD = 6` (125 ms), and
decoded positions/distances are plotted in cm (`BIN_CM`).

### Rollout sampling: a theta-skipping pacemaker gives left-right alternation

**The problem.** Each theta cycle is one stochastic rollout, and the cycles are
averaged into a Monte-Carlo estimate of the state distribution (Eq. 5). If the
cycles are drawn *independently*, the arm each sweep enters is an independent coin
flip, so the animal can easily waste several cycles in a row looking down the same
arm. The estimate is correct on average but noisy. Vollan et al. (2025) instead
find that real sweeps *alternate* left and right from cycle to cycle, on ≈ 80% of
triplets.

**The idea.** Alternation is what a standard Monte-Carlo variance-reduction trick
produces: **antithetic sampling**, i.e. deliberately negative dependence between
consecutive samples. Every sample is still individually valid, so the average
stays unbiased, but consecutive cycles cover the two arms more evenly than
independent draws would.

**The mechanism.** At a choice point the destination is chosen by inverse-CDF from
strata of width `p_j`, so a *single* uniform decides the branch. That uniform is
supplied by a **theta-skipping pacemaker** (`hippo/pacemaker.py`) — a phase locked
to half the theta frequency:

```
u^k = (u^0 + k/2 + ε^k) mod 1,     ε^k = ε^{k−1} + η^k
u^0 ~ Uniform[0,1),                η^k ~ N(0, σ_η²),     σ_η = 0.075
```

Branch `j` is chosen on cycle `k` iff `u^k` lands in its stratum. The `+1/2` drift
flips the branch every cycle (the antithetic coupling); the diffusion lets it
occasionally fail, which is what turns a perfect zigzag into the *partial*
alternation that is actually observed. Everything else about each rollout — dwell
times, the run down the arm — is drawn independently.

Two properties, both derived in [DERIVATIONS §C.11](DERIVATIONS.md):

| property | value |
|---|---|
| marginal consistency `P(X^k = j) = p_j` | exact, for every `k`, any `σ_η`, any `p_j` |
| adjacent-switch rate | `q(σ) = ½ + (4/π²) Σ_{n odd} n⁻² e^{−2π²n²σ²}` ≈ `1 − 1.596σ` |
| at `σ_η = 0.075` | `q = 0.880`, triplets 0.796 (observed 0.798) |
| variance of the branch estimate | ≈ 5.5× below independent sampling |

Marginal consistency holds because a uniform on the circle stays uniform under
convolution with any independent variable, so it is insensitive to `σ_η` — the
noise can be chosen purely to fit the alternation rate. The diffusion must
**accumulate**: i.i.d. jitter about a fixed phase satisfies the same marginal but
never forgets `u^0`, leaving a variance floor that does not shrink with the number
of cycles (§C.11.5).

**Consistency with the master equation.** Coupling the rollouts changes how they
are drawn, so it is worth being explicit about when the microscopic average still
reproduces the macroscopic dynamics. By linearity of expectation,
`E[(1/K)Σ z̃ₖ(t)] = z(t)` holds **iff each rollout is marginally a valid CTMC
path** — independence is never required, so any *coupling* is safe provided it
leaves the marginals alone ([DERIVATIONS §C.8](DERIVATIONS.md)). One scheme in
this repo deliberately does not satisfy the stronger *per-cycle* version:

| scheme | K-cycle average | single cycle |
|---|---|---|
| independent, antithetic (free order) | unbiased | unbiased |
| antithetic with `order_key` (used only by Fig. 1b's ideal limit) | unbiased | **biased** — the member is assigned to a cycle by its outcome |

Figure 1 is unaffected: the pacemaker above is exact *per cycle*, not merely on
average. The caveat applies only to the `order_key` route, which Fig. 1b uses to
measure the idealized limit, and there only `K`-cycle averages are read — but it
does mean one should not read a single theta cycle (or an odd-length window) from
that scheme. `test_micro_macro_consistency.py` checks all of this
against `exp((t/τ)W)` on five environments and asserts that expected failure, so
the caveat cannot silently regress.

[DERIVATIONS §C.9](DERIVATIONS.md) goes one step further and exhibits a mechanism
for which the marginal condition holds *by construction* rather than by
assumption: stratify the branch coordinate (so forcing a branch needs no symmetry
and no calibration) and choose the branch by **quota depletion** — give each
branch a quota `p_j·M` over a block of `M` cycles and select with probability
`(m_j − c_j)/(M − n + 1)`, i.e. use depletes a branch's own availability and the
block refills it. Exchangeability of the resulting random permutation makes every
cycle's marginal exactly `p_j`, and sampling without replacement makes the branch
frequency variance `p(1−p)/k · (M−k)/(M−1)` — below independent for every
`k ≥ 2` and exactly zero over a full block. `M = 2` *is* antithetic pairing. Like the
pacemaker, it is exact at the per-cycle level *and* antithetic; the `order_key`
route buys a perfect zigzag at the cost of the per-cycle caveat above, which is
why Figure 1 uses the pacemaker instead.

[DERIVATIONS §C.10](DERIVATIONS.md) then asks what circuit could run it, and finds
two. Note first that `Σⱼ(mⱼ−cⱼ) = M−(n−1)`, so the rule is just
`P[Xₙ=j] = Rⱼ/ΣₖRₖ` — **selection proportional to remaining resource, divisively
normalized by the total remaining**. (A) *A depleting synaptic resource*:
short-term synaptic depression supplies `Rⱼ`, the normalization pool supplies the
denominator, and a stochastic winner-take-all selects; exact if depletion is
quantal and resource is refilled to quota each block, only approximate under
ordinary exponential recovery. (B) *A slow ramp*: a phase advancing `1/M` per
theta cycle with a random offset per block, compared against the normalized
weights — provably exact per cycle, deterministically balanced, and needing one
random number per block rather than one per cycle. The branch law itself needs no
extra machinery: `pⱼ = W_cj/(−W_cc)` is the outgoing recurrent weight under
divisive normalization. Both are guarded in the test; they differ in whether an
externally imposed repeat suppresses that arm afterwards (A) or leaves the
sequence untouched (B). [DERIVATIONS §C.11](DERIVATIONS.md) analyses (B) in its
realistic form — a pacemaker whose phase *diffuses* — with closed forms for the
alternation rate `q(σ)` and for the variance of the Eq.-5 estimator, and the
reason the noise must accumulate rather than be i.i.d.

**Sweeps (panels A–D).** T-maze: 14-state stem to a choice point, two 8-state
arms. `λ = 0.2`, theta period `T = THETA_PERIOD`, 50 cycles decoded at 50 phase
bins, with the branch on each cycle set by the theta-skipping pacemaker
(`σ_η = 0.075`); the realized window alternates on 88% of adjacent cycles and 81%
of triplets, against a long-run 0.880 / 0.796 and an observed 0.798. Arm choice =
first arm state entered;
distance-ahead = graph distance from a stem launch point (state 10). Panel A
plots 16 *consecutive* cycles (not every other one, which under alternation would
show a single arm). Linear track: 60 states, 12 cycles, animal advancing 2.2
states/cycle (1D: no branch, so the pacemaker does not apply there). (`seed = 3`.)

**Phase precession (panels E–F).** Linear track, `N = 60`, target cell `f = 35`,
forward rate `k_forward = 1.0`, diffusion `k_back = 0.08` (drift 0.92, per-cycle
look-ahead `L ≈ 5.5`), `λ = 0.20` (default), sensory width `σ = 1.6`, theta period
`THETA_PERIOD`, 36 phase bins, 80 cycles/position. The cell spikes at a phase bin
with probability `exp(−(d−f)²/2σ²)` (`d` = decoded position); the phase-vs-
position slope (≈ −5.6 deg/cm) is a linear fit. Because the sweep runs forward,
the cell fires while the animal is behind `f` (predictive field, consistent with
Simulation 2). (`seed = 4`.)

---

## Simulation 2 — Predictive place-field structure and inferential reasoning (`sim2_predictive_inference.py`)

Three readouts of the same predictive map, one per figure column (schematic on
top, result below). The figure imports the backward-skew and reward-clustering
components (`sim2_backward_skew.py`, `sim2_reward_clustering.py`) and adds the
inference readout.

**A. Backward skew** (Mehta, Quirk & Wilson 2000: one-way running makes place
fields negatively skewed, shifts their center of mass backward, and expands
them). Linear track, `n = 40` states = 160 cm, run rightward; entorhinal input a
Gaussian tuning curve (`σ = 1.6` bins). `W` starts at zero; over `n_laps = 25`
laps `z` is clamped to the sensory bump and the target is the forward-shifted
bump `ż = (X[p+1]−X[p])/dt` (`dt = 1`); `W` updates by the three-factor rule
(`η = 0.08`), re-projected to a generator each step. Firing-rate map `F = X·R_T`
(truncated resolvent, one theta cycle) with `λ = 0.20` (default). The field grows
a backward tail with a sharp leading edge (skewness → −1.1, width ≈ 6 → 11 cm,
COM shifts ≈ 6 cm backward over 25 laps). (`seed = 0`.)

**B. Reward clustering** (Hollup et al. 2001: place fields accumulate at the goal
in an annular watermaze, with **randomized start positions**). Ring of `n = 72`
states (288-cm circumference), goal at index 0, `λ = 0.20`. We model a
**fully-trained agent with the optimal policy**: from any state it swims the
**shorter arc** to the goal (`drift = 1.3`) — so it goes clockwise from states on
one side of the goal and counter-clockwise from the other, *not* a single fixed
direction — with symmetric diffusion (`diffusion = 0.10`) and slowing near the
platform (rates scaled toward `dwell_factor = 0.40` over a Gaussian dwell width
`dwell_w = 6` states); the control is an undirected random walk. Field density =
**column sums of the truncated resolvent** — discounted future occupancy of each
location summed over **all** start states, i.e. the uniform/randomized start
distribution of the task — normalized so 1.0 = uniform. The finite horizon
sharpens the accumulation at the goal (peak ≈ 2.4× chance), shown as a polar plot.
The schematic shows two example optimal trajectories converging on the goal from
randomized starts (one clockwise, one counter-clockwise).

**C. Inferential reasoning** (Barron et al. 2020, *Cell*, "Neuronal Computation
Underlying Inferential Reasoning in Humans and Mice": presenting a cue reactivates
the representation of its inferred associate). `n = 12` triples of states —
audio `A_i`, visual `B_i`, outcome `C_i`. Preconditioning learns `A_i → B_i` and
conditioning learns `B_i → C_i` (transitions built into `W`, with per-triple
lognormal rate heterogeneity for a realistic spread), so the audio→outcome link
is only inferred (never co-experienced). `λ = 0.20`; the representation of a
stimulus is its row of the truncated successor representation `R_T`, and the
representational similarity between two stimuli is the correlation of these SR
patterns. Following Barron et al.'s Fig. 4F, the result is a **bar plot of
representational similarity by condition** — each audio cue vs its associated
visual (≈ 0.93), its inferred outcome (≈ 0.88), and non-associated stimuli (≈ 0);
bars are mean ± SEM over the `n` cues, dots the individual cues. Associated and
inferred both far exceed non-associated — the transitive resolvent activates the
inferred representation.

---

## Simulation 3 — Multi-scale place fields and discounting (`sim3_multiscale_discounting.py`)

A single leak-rate distribution, taken from CA1 data — `λ = 1/τ` for the 32 cells
best fit by exponential decay (`CA1_TAU`, from doi:10.1002/hipo.70105, SI). The
time constants `τ` are **exponentially distributed** (KS *p* = 0.91), mean
`μ = mean(1/τ) ≈ 0.19 s⁻¹` — read out spatially (A–C) and temporally (D–E) from
the same population.

**Spatial — place-field widths (A–C).** This is the repository's only
**encoding-mode** read-out: CA3 is clamped to EC input, so CA1 integrates the
animal's *experienced* position with its own slow leak rate (`CA1_TAU`,
τ = 1–49 s) over behavioural time rather than accumulating a theta rollout.
The horizon is therefore not a theta cycle — `THETA_PERIOD` does not appear in
the file — and the intrinsic tuning width is the *asymptotic* (steady-state)
spatial scale, so it uses the full resolvent. Composing it instead with the
one-theta-cycle kernel would impose a floor on the width equal to one sweep's
reach (12.4 cm RMS, below which 67% of recorded fields lie) and would break the
`w = c/√λ` law (slope −0.18 vs −0.50); see `check_two_timescale.py`. Panel A shows
three example resolvent fields (2D open arena, `52 × 44` bins = 208×176 cm) at
6 / 14 / 27 cm, cropped to a **common** ±72 cm window and rendered on a **log**
colour scale floored at 10⁻³ of peak (`FIELD_CROP_CM`, `FIELD_VMIN`). Both
choices are needed to make the widths visible: the resolvent column is cusped at
the source (it falls to 0.21 / 0.38 / 0.49 of peak within one 4 cm bin, so the
half-max radius is one bin at *every* leak rate), which means the width is
carried by the exponential tail alone — a linear colour scale over the whole
208×176 cm arena renders all three as near-identical dots. The dashed ring marks
the measured RMS width, i.e. exactly the quantity plotted in Panel B; Panel B verifies that the measured widths follow `c/√λ` (log–log,
one calibration scale `c`, with the expected droop where the largest fields reach
the arena wall). **Panel C** compares the *aggregated empirical* tuning-width
distribution — Cohen & Drugowitsch (2025), Pfeiffer & Foster (2013) dataset, all
8 sessions pooled, `n = 32 000`, in `field_widths/field_widths.pkl` — to the
model's prediction: the pushforward of the leak-rate distribution through
`w = c/√λ`. The *shape* is derived, not fit — only the single scale `c` is a free
parameter — and the Panel-C inset and the width curve are held strictly
consistent (the width curve is the pushforward of the fitted inset density). The
inset fits the leak rate `λ = 1/τ` with an **inverse-exponential** density (i.e.
the *time constants* `τ = 1/λ` are exponential, KS *p* = 0.91); its pushforward
`w = c·√τ` is a **Rayleigh / Weibull-2** law. (Fitting a plain exponential to `λ`
instead would push forward to a Fréchet-2 that fits the widths worse — KS 0.10 vs
0.02, with too sharp a peak and a tail p95 ≈ 37 cm vs the data's 22; see
`analysis_width_leakrate.py` — so the inverse-exponential is used.) With the
single scale
`c ≈ 3.3 cm·s^½` calibrated to the empirical median it also predicts the mean and
upper percentiles (median 10.1, mean 10.7, p95 21.0 cm vs empirical 10.1 / 10.9 /
22.3).

**Temporal — discounting (D–E).** A bank drawn from the same leak-rate
distribution (`λ = 1/τ`, `τ ∼ Exp`) discounts as its Laplace transform,
`D(t) = E[e^{−λt}]` — approximately hyperbolic `1/(1+μt)` (max abs. deviation
0.033) and far more sustained than any single exponential `e^{−μt}` (times in
seconds). Intertemporal choice: smaller-sooner (amount 1.0, delay `0.5/μ`) vs
larger-later (1.6, delay `2.0/μ`), vs how far in advance the choice is made.
(`analysis_time_constants.py` plots the same `λ = 1/τ` data standalone.)

---

## Simulation 4 — DG-gated context states and memory retrieval (`sim4_context_dg.py`)

**Experiment.** Coelho et al. (2024), *Science Advances* 10:eadn9815. The three
panels mirror specific panels of that study (cited below).

**Environment.** Shared ring of `n = 30` states (120 cm), `λ = 0.20` (default),
`seed = 3`, Gaussian bump inputs (`σ = 1.6`).

**Contexts.** Two sparse, anti-correlated DG gating patterns (`frac = 0.6` ON at
gate 1, rest at `lo = 0.05`): a shared core plus context-specific cells. Context
vectors `c_A = (1,0)`, `c_B = (0,1)`; a balanced input `(0.5,0.5)` is the
DG-silenced / ambiguous case.

**Panels.**
- **A — context-specific neural states** (their **Fig. 3H–J**): the CA1
  population-vector correlation matrix under the two intact DG contexts (2N×2N)
  shows high within-context and low across-context correlation (within 0.91 vs
  between 0.44) — the two contexts are distinct neural states.
- **B — DG reactivation reinstates the state** (their **Fig. 5D–E**): under an
  ambiguous (uniform) cue, the DG input is swept `c = (1−α)c_A + α c_B`; CA1 state
  similarity to the pure context-A/context-B templates crosses over as the driven
  ensemble goes A → balanced → B. The `α = 0.5` midpoint is the DG-silenced case.
- **C — context-appropriate retrieval** (their **Fig. 2D–E, F–G**): retrieval
  probability of the context-A association is a softmax (`β = 3.5`) over the two
  template similarities. It is ≈1 when context A's ensemble is driven, ≈0 for B's,
  and ≈0.5 at balanced/silenced DG — the neural + behavioural analog of their
  **Fig. 1G–H**, where silencing DG made mice dig equally in both wells.

---

## Simulation 5 — Partial vs complete remapping (`sim5_partial_remapping.py`)

**Experiments.** Leutgeb et al. (2005, partial/gradual); Wills et al. (2005,
complete/abrupt). Per **Sanders, Wilson & Gershman (2020)** these differ because
of the training **data** — specifically the amount of training before the morph
was tested (among other protocol differences).

**Mechanism — context differentiation with training (no added machinery).** We
make one minimal assumption: the two stored context representations become
increasingly **differentiated** with training, summarized by a scalar
differentiation `g ≥ 0` that grows monotonically with the amount of training. A
morph cue `m ∈ [0,1]` (square `m = 0` → circle `m = 1`) is assigned to the two
contexts by similarity,
`s(m) = logistic(g·(2m−1))`, `c(m) = [1−s(m), s(m)]` (`context_vector`),
so as the representations separate the assignment sharpens on its own. A *single*
`g` sets both signatures of the transition — endpoint separation
`|s(1)−s(0)| = tanh(g/2)` (the degree of remapping) and midpoint slope
`s′(½) = g/2` (its abruptness) — both rising with `g`. `c(m)` gates CA3 exactly as
in Simulation 4 (same ring: `n = 30`, `λ = 0.20`, two overlapping ensembles
`frac = 0.6`, `lo = 0.05`); the ensemble overlap is the shared, non-remapping
core. `seed = 2`. No latent-state inference, competition temperature, or other
extra parameter is introduced — the only assumption is that `g` increases with
training. Differentiation levels: `DIFF_LO = 1.0` (matching Leutgeb),
`DIFF_HI = 6.0` (matching Wills).

**Panels.** *A* — morph response (CA1 similarity to the square map across 21 morph
levels) at low differentiation (`g = 1` → gradual, partial remapping, similarity
≈ 0.85 at the circle end, matching Leutgeb) vs high differentiation (`g = 6` →
abrupt, complete remapping, similarity ≈ 0.48, matching Wills). The curves are
labelled by differentiation, not by training amount: the two studies were not a
controlled early-vs-late comparison. *B* — remapping between the extremes
(1 − square/circle correlation) versus differentiation `g` (`0 … 6.6`); it grows
from ≈ 0 and saturates ≈ 0.52 (the shared core caps it below 1). *C* — per-cell
normalized activity across the morph at high differentiation (cells sorted): ≈ 67 % of cells
switch at the transition while a shared core (cells in both ensembles) persists —
partial remapping at the population level.

---

### Panel D — experienced morph frequency (`sim5b_morph_frequency.py`)

**Experiment.** Plitt & Giocomo (2021, *Nat. Neurosci.* 24:705–714). A VR
environment morphed between two endpoints at values `0, 0.25, 0.5, 0.75, 1` with
±0.1 per-feature jitter. Two groups differed only in how often intermediate
morphs occurred: **Rare Morph** (4 mice) < 4% of trials; **Frequent Morph**
(5 mice) 60%. Rare-Morph mice switched representation coherently at a threshold
morph; Frequent-Morph mice changed gradually (their Fig. 2, 3f–g;
KS `D = 0.395` between the Similarity Fraction distributions).

**Network.** Identical to panels A–C — same ring (`n = 30`, `λ = 0.20`), same
two overlapping DG ensembles (`frac = 0.6`, `lo = 0.05`), same truncated-resolvent
read-out, `seed = 2`. The components are imported from
`sim5_partial_remapping.py`, not reimplemented.

**What differs between groups: only the prior.** The DG context is the posterior
mean over a latent context `s`,

```
prior       π(s)   = experienced morph frequencies, each smeared by the jitter
                     (Gaussian, σ_jit = 0.10)
likelihood  p(m|s) = N(m; s, σ_obs²),   σ_obs = 0.20
context     c(m)   = [1 − E[s|m], E[s|m]]
```

Group weights are taken straight from the paper: Rare = `(0.48, 0.013, 0.013,
0.013, 0.48)`, Frequent = `(0.2, 0.2, 0.2, 0.2, 0.2)` — 60% over three
intermediates is 20% each, so the Frequent group's experience is exactly uniform,
which is the flat prior of their model.

**Mechanism.** Under a flat prior the posterior mean tracks the cue,
`E[s|m] ≈ m`, essentially independently of `σ_obs` (measured slope 0.98, 0.97,
0.79 at `σ_obs = 0.1, 0.2, 0.3`). Under a bimodal prior an ambiguous cue is
resolved by the prior, which places all its mass at the endpoints, so the code
switches categorically (slope 1.53, 3.13, 2.11 at the same `σ_obs`). **Sensory
uncertainty is what converts a bimodal prior into a step-like code** — with a
precise cue (`σ_obs → 0`) even the Rare group would represent intermediates
veridically. `σ_obs = 0.20` maximizes the slope dissociation (3.1 vs 1.0).

**What panel D plots.** CA1 similarity to the 0-end map across the morph for the
two groups: step-like (Rare, slope 3.1) vs near-linear (Frequent, slope 1.0).
This module is a *component* — it produces no figure of its own. Run it directly
and it reports the rest of the comparison, which the combined figure does not
plot: the Similarity Fraction over 4,000 simulated trials (morph drawn uniformly
from the five values, plus jitter), rescaled so the endpoints read 0 and 1
because the shared DG core keeps them correlated. Rare puts only 13% of trials
in the middle band against 42% for Frequent — bimodal vs unimodal, as reported.

**Honest limits.** The dissociation is robust (`D` significant at every `σ_obs`
tested) but smaller than reported: `D = 0.22` at the default, peaking at ≈ 0.28,
against 0.395. Loosening the DG overlap does not change `D` (it is flat at 0.225
for `frac = 0.6…0.9`), so the gap is not the shared core but the fact that a
two-ensemble code cannot remap as completely as a real CA1 population. The
simulation reproduces the form and direction of their result, not its exact
effect size.

---

## Simulation 6 — Splitter cells (`sim6_splitter_cells.py`)

**Experiments.** Wood et al. (2000); Frank et al. (2000).

**Environment.** Figure-eight maze (`figure_eight`): shared central stem of
`stem = 12` states (48 cm) to a choice point, then left and right return loops of
`loop = 16` states each back to the stem base. `λ = 0.20` (default), `seed = 4`. Stem
position is plotted in cm.

**Trajectory contexts.** Two DG-gated contexts (left-, right-bound). Each applies
an anti-correlated smooth per-cell gain (`gL = clip(1 + 0.8h)`,
`gR = clip(1 − 0.8h)` for a smooth random field `h`); the field takes both signs
across the stem, so some stem cells prefer left trials and others right (`seed =
4` gives a balanced mix). Each context also routes the correct return loop (the
opposite loop's columns gated to `lo = 0.05`).

**Analysis.** "Stem cells" are those whose field peaks on the stem. In-field rate
= `R[peak, cell]` on each trajectory; split index = `|L−R|/max(L,R)` (splitter
above 0.2). The figure shows the maze schematic (A) and three example stem cells
spanning both preferences (B; two fire more on left trials, one on right — the
example picker takes the strongest left-preferring, the strongest right-
preferring, and the next strongest splitter). Population statistics — mean split
≈ 0.35, ≈ 67% splitters, and trajectory selectivity `1 − corr(R_L[p,:], R_R[p,:])`
rising ≈ 0 → 0.9 toward the choice point (prospective coding) — are still computed
and printed but no longer plotted as separate panels. Prospective coding emerges from the predictive map
alone; trajectory-dependent rate on the stem comes from the context gain.
