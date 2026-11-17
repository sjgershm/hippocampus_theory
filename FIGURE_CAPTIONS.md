# Figure captions

Panel-by-panel captions for every figure in [`figures/`](figures/) (saved as
PDF). Each figure is produced by the correspondingly numbered script in
[`sims/`](sims/); panel labels (bold, top-left) and short centered titles are
added at save time. Equation numbers refer to the note (*"Prediction,
imagination, memory: variations on a hippocampal theme"*). Space is in physical
cm (`BIN_CM = 4 cm` per bin, applied consistently); one theta cycle is 125 ms
(`THETA_PERIOD`).

Two read-out modes appear. In **simulation mode** CA3 rolls out and CA1
accumulates a single sweep, so the operator is the **truncated resolvent**
(Eq. 14) at the fast within-cycle rate, over one theta cycle; this covers every
figure below except one. In **encoding mode** CA3 is clamped to EC input and
CA1 integrates the animal's *experienced* position with its own slow leak rate
over behavioural time, so the operator is the asymptotic resolvent and no theta
horizon enters; this is Fig. 3A–C (the intrinsic tuning width) and nothing
else. Stated once here, the individual captions below do not repeat it.

The figures are ordered by theme: imagination/prediction dynamics (theta sweeps
and phase precession, Fig. 1); predictive place-field structure and inference
(backward skew, reward clustering, inferential reasoning, Fig. 2); the
multi-scale predictive map (place-field widths and temporal discounting from one
leak-rate distribution, Fig. 3); and context, memory and trajectory coding (DG
gating, remapping, splitter cells, Figs. 4–6).

---

## Figure 1 — Forward sweeps and phase precession (`sim1_theta_phase.pdf`)

Both phenomena come from one mechanism (Eq. 5): each theta cycle is a forward
rollout of the generator from the animal's current state, so the decoded
position sweeps ahead within the cycle.

The branch taken on each theta cycle is set by a **theta-skipping pacemaker**
(`hippo/pacemaker.py`): a phase locked to half the theta frequency that drifts by
`1/2` each cycle and diffuses slowly,
`u^k = (u^0 + k/2 + ε^k) mod 1` with `ε^k = ε^{k−1} + η^k`, `η ~ N(0, σ_η²)`,
`σ_η = 0.075`. The drift flips the branch every cycle — this is the antithetic
coupling — while the diffusion lets it occasionally fail, reproducing the *partial*
alternation reported by Vollan et al. (2025, *Nature* 639:995–1005) rather than a
perfect zigzag. Because a uniform on the circle stays uniform under convolution
with any independent variable, `u^k` is marginally uniform at every cycle, so each
rollout remains an exact draw of the generator and the Eq.-5 average stays
unbiased ([DERIVATIONS §C.11](DERIVATIONS.md)); the coupling only removes sampling
noise. Fig. 1b quantifies that gain for the *idealized* coupling (`σ_η → 0`,
exact antithetic pairs); at `σ_η = 0.075` the branch estimate is ≈ 5.5× sharper
than independent sampling ([DERIVATIONS §C.11.4](DERIVATIONS.md)).

- **(A)** *Sweeps into both arms.* T-maze (gray corridors) with the animal at the
  choice point (triangle); decoded rollout sweeps from successive cycles, colored
  by the arm entered (blue = left, red = right), run forward into both arms.
- **(B)** *Forward sweep per cycle.* Decoded distance ahead of the animal (cm)
  versus theta phase (gray = individual cycles, orange = mean); the
  representation sweeps forward within each cycle.
- **(C)** *Left-right alternation.* Which arm each successive theta cycle's sweep
  ran down. **One dot per theta cycle** (50 consecutive cycles, left to right):
  its vertical position is the arm that cycle's rollout entered — bottom row
  (blue) = left arm, top row (red) = right arm — and the thin gray line simply
  joins consecutive cycles so the alternation is visible as a zigzag. There is no
  continuous quantity here; the two rows are the only two values the variable can
  take. The pacemaker's half-cycle drift makes the arm flip from cycle to cycle,
  and the breaks in the zigzag are cycles where its diffusing phase failed to
  cross the branch boundary: 88% of adjacent cycles switch and 81% of triplets
  alternate, against the 79.8% of triplets reported by Vollan et al. Under
  independent sampling the same plot would instead be a coin flip per cycle: ~50%
  switches, ~25% of triplets alternating, and long runs on one arm.
- **(D)** *Theta sequences.* On a linear track, decoded position over twelve
  cycles as the animal advances (dashed = true position); each cycle resets near
  the animal and sweeps ahead.
- **(E)** *Theta sweeps vs field.* Decoded position versus theta phase for
  rollouts at three animal positions approaching a place field (dashed = field
  location f); the closer the animal, the earlier in the cycle the sweep reaches
  f.
- **(F)** *Phase precession.* Spike theta phase versus animal position (two
  cycles, 0–720°; red = linear fit). Phase precesses (negative slope) across the
  field.

---

## Figure 1b — Alternating sweeps sharpen the predictive estimate (`antithetic_variance.pdf`)

Produced by [`analysis_antithetic_variance.py`](analysis_antithetic_variance.py);
the normative counterpart to the alternation in Fig. 1C. Why should sweeps
alternate? Because alternation *is* antithetic sampling, and antithetic sampling
buys precision for free. Independent sweeps (gray) vs antithetic pairs (red) from
the choice point of the T-maze, with variance measured across 2,000 independent
replicates of the `K`-cycle estimator of Eq. 5.

- **(A)** *Unbiased, but far less variable.* Estimated time-averaged occupancy of
  the left arm versus the number of cycles aggregated `K` (line = mean over
  replicates, band = ±1 SD, dashed = the exact value from `exp(tW)`). Both schemes
  are centered on the truth — antithetic pairing introduces **no bias** — but the
  antithetic band is dramatically tighter. After a single pair of cycles the
  independent estimate is still scattered over most of the interval.
- **(B)** *Variance falls as 1/K.* Variance of the same estimate (log–log). Both
  schemes scale as `1/K`; antithetic pairing simply sits ≈ 47× lower, i.e. it
  reaches a given precision with ≈ 47× fewer theta cycles.
- **(C)** *Which estimates get sharper.* Variance reduction **factor** — the
  *ratio* `Var(independent) / Var(antithetic)`, not a difference, so 1 means no
  gain — at `K = 16` for six estimators built from the *same* sweeps. The gain is **structured**, not uniform: the branch probability
  `P(left)` becomes *exact* (zero variance), arm occupancy improves ≈ 47×, and the
  look-ahead distance ≈ 1.9× (that one comes from the antithetically coupled dwell
  times). The fine-grained belief `z(t)` gains only ≈ 1.2×, because within-arm
  timing noise is untouched and the two members of a pair occupy disjoint arms, so
  their fluctuations cannot cancel. The left-minus-right value *contrast* gains
  nothing: for a quantity that is *antisymmetric* across the branch, the negative
  coupling works against the balancing it provides.
- **(D)** *Cumulative coverage.* Space covered (cm) by `K` aggregated sweeps.
  Alternation covers more ground early (50 vs 41 cm after two cycles) — the
  efficient "look around" argued for by Vollan et al. (2025).

---

## Figure 2 — Predictive place-field structure and inferential reasoning (`sim2_predictive_inference.pdf`)

Three readouts of the same predictive map (the truncated resolvent / successor
representation), one per column: a task schematic on top, the model result below.

- **(A)** *Backward skew* (Mehta et al. 2000). *Schematic:* one-way running on a
  linear track. *Result:* firing-rate profile of a central cell at five stages of
  learning (laps 1, 3, 6, 12, 25), normalized and aligned to the field peak. As
  the recurrent generator `W` is learned online (Eq. 21), the field starts
  symmetric and grows a tail extending backward (opposite the direction of
  travel), with a sharp leading edge.
- **(B)** *Reward clustering* (Hollup et al. 2001). *Schematic:* annular watermaze
  with a hidden goal; a fully-trained agent takes the shorter arc to the goal from
  randomized start positions (dots), so trajectories converge on it from both
  sides — clockwise from one side, counter-clockwise from the other. *Result:*
  polar plot of the ring, each sector colored by predicted field density (column
  sums of the resolvent, i.e. occupancy summed over all/randomized start
  positions) under the optimal goal-directed policy; density is highest at and
  around the goal (star).
- **(C)** *Inferential reasoning* (Barron et al. 2020, *Cell*). *Schematic:* an
  associative-inference chain — audio→visual and visual→outcome are learned
  directly, audio→outcome is only inferred. *Result:* representational similarity
  between each auditory cue's evoked pattern (its successor-representation row)
  and its **associated** visual cue, its **inferred** outcome, and
  **non-associated** stimuli (bars = mean over cues, error bars = SEM, dots =
  individual cues). Associated (≈ 0.93) and inferred (≈ 0.88) are both high while
  non-associated is ≈ 0 — the cue reactivates its associated *and* its inferred
  (never-paired) representation, as in the mouse hippocampal RSA (cf. Barron
  et al. Fig. 4F).

---

## Figure 3 — Multi-scale place fields and discounting (`sim3_multiscale_discounting.pdf`)

A single population of CA1 cells whose time constants `τ = 1/λ` are exponentially
distributed (doi:10.1002/hipo.70105, KS *p* = 0.91), so the leak rates `λ` follow
an inverse-exponential law (C inset) — read out two ways: spatially as
place-field widths (A–C, since intrinsic width `= c/√λ`) and temporally as the
discount function (D–E, its Laplace transform).

**Encoding mode** (see the header): this is the only encoding-mode figure, which
is why the operator is the asymptotic resolvent and `THETA_PERIOD` appears
nowhere in the simulation. The `w = c/√λ` law is exact in the large-arena
limit — fitted log–log slopes converge −0.4888 → −0.4965 → −0.4992 for arenas
of 144×120, 208×176 (the one used) and 272×232 cm, with `c` stable at
8.0 cm·√unit and *r* = −0.99997 (`check_two_timescale.py`).

- **(A)** Model place fields at three widths (≈ 6, 14, 27 cm), all shown in the
  **same** ±72 cm window (scale bar 50 cm) so the sizes are directly comparable;
  field size grows as the leak rate falls. Colour is activity as a fraction of
  peak on a **log** scale. The log scale matters: the resolvent column has a
  sharp cusp at the source — it drops to 0.21 / 0.38 / 0.49 of peak within one
  4 cm bin, so its half-max radius is a single bin at *every* leak rate, and the
  width lives entirely in the exponential tail. On a linear scale that tail is
  crushed into the bottom of the colour range and the three fields look alike.
  The dashed ring is the measured RMS width — the quantity plotted in (B).
- **(B)** *Field width vs leak rate.* Measured widths (cm) versus leak rate
  `λ` (s⁻¹, log–log) fall on the derived `w = c/√λ` line (single scale `c`); the
  droop at the largest fields is the arena boundary.
- **(C)** *Width distribution.* Aggregated **empirical** CA1 tuning widths (cm;
  Cohen & Drugowitsch 2025 / Pfeiffer & Foster 2013, 8 sessions pooled, green)
  versus the model **prediction** (blue): the pushforward of the leak-rate
  distribution (inset, `λ = 1/τ`, with its inverse-exponential fit) through
  `w = c/√λ` — inset and prediction are the *same law*. The *shape* is derived —
  only the single scale `c` is calibrated — and because the time constants
  `τ = 1/λ` are exponential it is a Rayleigh/Weibull-2 law that matches the median,
  mean and upper percentiles (KS 0.02). (A plain exponential fit to `λ` would
  instead push forward to a heavier Fréchet tail that fits the widths worse —
  KS 0.10, p95 ≈ 37 vs 22 cm; see `analysis_width_leakrate.py` and
  [DERIVATIONS §3.7](DERIVATIONS.md).)
- **(D)** *Hyperbolic discounting.* Discount factor vs delay (s): a single leak
  rate is exponential (orange); the full bank discounts as the Laplace transform
  of the same leak-rate distribution — approximately hyperbolic `1/(1+μt)` (blue
  over black dashed) and far more sustained than any single timescale.
- **(E)** *Preference reversal.* Discounted value of a smaller-sooner vs a
  larger-later reward as a function of how far in advance the choice is made, for
  single-exponential (dotted) vs the bank (solid); the bank produces a preference
  reversal (marked), which no single exponential can.

---

## Figure 4 — DG gates context-dependent neural states and memory retrieval (`sim4_context_dg.pdf`)

Simulates **Coelho et al. (2024)**, *Science Advances* 10:eadn9815. The DG context
vector `c` gates the CA3 generator through `G` (Eqs. 17–18); the CA1 read-out is
the context-gated truncated (one-theta-cycle) resolvent. Panel letters cite the
specific panels of the original study.

- **(A)** *Context-specific neural states* — simulates the paper's **Fig. 3H–J**.
  CA1 population-vector correlation matrix under the two intact DG contexts
  (2N × 2N): high within-context blocks and low across-context blocks mean the
  two contexts settle into distinct neural states (their between/within
  population-vector distance).
- **(B)** *DG reactivation reinstates the state* — simulates the paper's
  **Fig. 5D–E**. Given an ambiguous cue, CA1 state similarity to the context-A
  and context-B templates as the *driven* DG ensemble is swept from A to B;
  driving a context's ensemble makes the CA1 state resemble that context. The
  balanced midpoint (no DG bias) is the ambiguous / DG-silenced condition of
  their **Fig. 1**.
- **(C)** *Context-appropriate memory retrieval* — simulates the paper's
  **Fig. 2D–E, F–G** (its headline result). Probability of retrieving the
  context-A association versus the driven DG ensemble; reactivating a context's
  ensemble biases retrieval to that context's association, whereas a
  balanced/silenced DG gives no preference (their **Fig. 1G–H**, where silencing
  DG made mice dig equally in both wells).

---

## Figure 5 — Partial vs complete remapping in morphed environments (`sim5_partial_remapping.pdf`)

An environment morphed square → circle (Leutgeb et al. 2005: partial/gradual;
Wills et al. 2005: complete/abrupt). Per Sanders, Wilson & Gershman (2020) the
difference reflects the training **data** and protocol, not a model parameter
(the two studies differed in several respects and were not a controlled
comparison). We capture this with a single, minimal assumption and no added
machinery: the two stored context representations become increasingly
**differentiated** with training (`g`). A morph cue `m ∈ [0,1]` is assigned to
the two contexts by similarity, `s(m) = logistic(g·(2m−1))`,
`c(m) = [1−s(m), s(m)]`, so as the representations separate the assignment — and
hence the remapping — sharpens on its own (endpoint separation `tanh(g/2)` and
midpoint slope `g/2` both grow with `g`). `c` gates CA3 (Eqs. 17–18); the two DG
ensembles overlap, leaving a shared, non-remapping core. Small `g` → gradual,
partial remapping (matching Leutgeb); large `g` → abrupt, complete remapping
(matching Wills). The same network gives both, differing only in `g`.

- **(A)** *Gradual vs abrupt.* CA1 map similarity to the square map across the
  morph, at low differentiation (`g = 1` → gradual, partial remapping — matching
  Leutgeb) vs high differentiation (`g = 6` → abrupt, complete remapping —
  matching Wills). The same network gives both, differing only in `g`.
- **(B)** *Remapping grows with differentiation.* Remapping between the two
  extremes (1 − square/circle correlation) versus context differentiation `g`. It
  saturates below 1 because the overlapping ensembles keep a shared core.
- **(C)** *Partial remapping.* Per-cell normalized activity across the morph
  (cells sorted); most cells switch at the transition while a shared core (cells
  in both ensembles) does not — partial remapping at the population level.
- **(D)** *Experienced morph frequency.* The same sharp-vs-gradual contrast,
  produced by a **manipulated** variable rather than by `g`. Simulates **Plitt &
  Giocomo (2021)**, *Nat. Neurosci.* 24:705–714, who morphed a VR environment
  between two endpoints while varying how often intermediate morphs occurred: a
  **Rare Morph** group saw < 4% of trials at intermediate morphs, a **Frequent
  Morph** group 60%. Rare-Morph mice held one representation up to a threshold
  morph and then switched coherently; Frequent-Morph mice changed gradually.
  The network here is unchanged from panels A–C — the *only* difference between
  the two curves is the prior over the latent context `s` that each group's
  experience induces. The DG context is the posterior mean,
  `c(m) = [1 − E[s|m], E[s|m]]`, with a Gaussian likelihood of width
  `σ_obs = 0.20`. **Why the manipulation works:** under a flat prior (Frequent,
  whose 60% over three intermediates is 20% each, i.e. uniform) the posterior
  mean tracks the cue veridically, `E[s|m] ≈ m`, whatever the sensory noise;
  under a bimodal prior (Rare) an ambiguous cue is resolved by falling back on
  the prior, which says "it is one of the two endpoints", so the code switches
  categorically. Sensory *uncertainty* is therefore what converts a bimodal
  prior into a step-like code. Transition slopes 3.1 (Rare) vs 1.0 (Frequent).
  As in panel A, similarity does not fall to zero because the two ensembles
  share a core. Running `sims/sim5b_morph_frequency.py` directly also reports
  the Similarity-Fraction comparison this panel does not plot: 13% of
  Rare-Morph trials fall in the middle band against 42% for Frequent
  (bimodal vs unimodal, as reported), with KS `D = 0.22` here against their
  0.395 — the right direction and form, a smaller effect, because a
  two-ensemble code cannot remap as completely as a real CA1 population.

---

## Figure 6 — Splitter cells on a continuous-alternation maze (`sim6_splitter_cells.pdf`)

A figure-eight maze with a shared central stem; two trajectory types are modeled
as two DG-gated contexts (each routes the correct return loop and applies an
anti-correlated gain).

- **(A)** *Alternation maze.* Schematic: shared stem (black), left (blue) and
  right (red) return loops, choice point (triangle).
- **(B)** *Splitter cells.* Firing rate along the shared stem (cm) on left (blue)
  vs right (red) trials for three example cells (normalized, offset); each keeps
  its field location but fires at a trajectory-dependent rate. The examples span
  both preferences — some cells fire more on left trials, others on right.

---

## Auxiliary — Fréchet vs Gamma (`frechet_vs_gamma.pdf`)

Produced by [`analysis_frechet_vs_gamma.py`](analysis_frechet_vs_gamma.py) for
[DERIVATIONS §3.7](DERIVATIONS.md). This concerns the *alternative* exponential-
*leak-rate* assumption, whose width pushforward is Fréchet-2 (the data-matching
prediction in Fig. 3C is instead the lighter Rayleigh implied by the exponential-
*τ* CA1 data). Fréchet-2 widths with a Gamma fit and a Fréchet fit overlaid, on a
linear axis (left: the bodies coincide) and a log axis (right: the tails diverge —
Fréchet power-law vs Gamma exponential, where the data are sparse).
