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

**Experiments.** Johnson & Redish (2007); O'Keefe & Recce (1993); Skaggs et al.
(1996). One mechanism: each theta cycle is a forward rollout (Eq. 5).

All sweeps are decoded over one theta cycle, `T = THETA_PERIOD = 6` (125 ms), and
decoded positions/distances are plotted in cm (`BIN_CM`).

**Sweeps (panels A–D).** T-maze: 14-state stem to a choice point, two 8-state
arms. `λ = 0.2`, theta period `T = THETA_PERIOD`, 50 cycles decoded at 50 phase
bins; arm choice = first arm state entered; distance-ahead = graph distance from
a stem launch point (state 10). Linear track: 60 states, 12 cycles, animal
advancing 2.2 states/cycle. (`seed = 3`.)

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

**Spatial — place-field widths (A–C).** The intrinsic tuning width is the
*asymptotic* (steady-state) spatial scale, so it uses the full resolvent. Panel A shows
three example resolvent fields (2D open arena, `52 × 44` bins = 208×176 cm) at
6 / 14 / 28 cm; Panel B verifies that the measured widths follow `c/√λ` (log–log,
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
