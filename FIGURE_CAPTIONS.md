# Figure captions

Panel-by-panel captions for every figure in [`figures/`](figures/) (saved as
PDF). Each figure is produced by the correspondingly numbered script in
[`sims/`](sims/); panel labels (bold, top-left) and short centered titles are
added at save time. Equation numbers refer to the note (*"Prediction,
imagination, memory: variations on a hippocampal theme"*). Space is in physical
cm (`BIN_CM = 4 cm` per bin, applied consistently); one theta cycle is 125 ms
(`THETA_PERIOD`). All behavioural/predictive-map readouts use the **truncated
resolvent** (Eq. 14) — the successor representation over a single theta sweep —
except the intrinsic tuning width (Fig. 3A–C), which is the asymptotic `c/√λ`
scale.

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

- **(A)** *Sweeps into both arms.* T-maze (gray corridors) with the animal at the
  choice point (triangle); decoded rollout sweeps from successive cycles, colored
  by the arm entered (blue = left, red = right), run forward into both arms.
- **(B)** *Forward sweep per cycle.* Decoded distance ahead of the animal (cm)
  versus theta phase (gray = individual cycles, orange = mean); the
  representation sweeps forward within each cycle.
- **(C)** *Serial arm sampling.* Arm represented on successive theta cycles
  (blue = left, red = right); the two arms are sampled serially across cycles.
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

- **(A)** Model place fields at three widths (≈ 6, 14, 28 cm) as 2D rate maps
  (208×176 cm arena); field size grows as the leak rate falls.
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
