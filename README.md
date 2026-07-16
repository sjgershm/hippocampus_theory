# Variations on a hippocampal theme — model & simulations

Python implementation of the network model in **Gershman, *"Prediction,
imagination, memory: variations on a hippocampal theme"*** and a set of
illustrative simulations that recover well-known empirical phenomena from a
single circuit.

## The model

A linear recurrent network with three populations:

| symbol | population | role |
|--------|------------|------|
| `x`    | entorhinal cortex (EC) | input / belief state |
| `z(t)` | CA3 | recurrent internal state |
| `y(t)` | CA1 | leaky-integrator output |

with dynamics (note Eqs. 1–2, treating `z`, `y` as row vectors)

```
tau * dz/dt =  z W                 # CA3 recurrent dynamics
tau * dy/dt = -lam y + z           # CA1 leaky integration
```

`W` is the **generator** of a continuous-time Markov chain (master equation),
so `z(t) = x · exp((t/tau) W)` is the *propagator* applied to the input — a
form of imagination about the future. The model's key objects, all in
[`hippo/model.py`](hippo/model.py):

| object | code | note |
|--------|------|------|
| propagator `exp((t/τ)W)` | `propagator` | imagination / rollouts |
| resolvent `(λI−W)⁻¹` | `resolvent` | continuous-time **successor representation** (Eq. 8) |
| truncated resolvent `∫₀ᵀe^{−λt}e^{Wt}dt` | `resolvent_truncated` | SR over one theta sweep (Eq. 14) |
| value `x(λI−W)⁻¹r` | `value` | discounted return (Eq. 13) |
| Laplace bank `{(λₘI−W)⁻¹}` | `laplace_bank` | multi-scale predictive map (Eq. 16) |
| stochastic rollout | `sample_trajectory` / `decode_sweep` | one coherent theta-cycle sweep (Eq. 5) |
| eigen-decomposition | `spectrum` | spectral analysis (Eqs. 6–7) |
| three-factor learning | `learn_generator` | online generator rule (Eq. 21) |
| DG context gating | `ContextHippocampalModel` | `J_ij = W_ij b_j`, `b = cG` (Eqs. 17–18) |

The leak rate `λ` is the continuous-time analog of the RL discount factor: a
larger `λ` discounts the future more steeply and yields smaller / shorter-range
place fields.

**Physical units.** Space is `BIN_CM = 4 cm` per bin and one theta cycle is
`THETA_PERIOD` (125 ms), applied consistently across all simulations (axes in
cm). Every behavioural/predictive-map readout uses the **truncated resolvent**
`R_T = (λI−W)⁻¹(I − e^{−(λI−W)T})` (Eq. 14, `resolvent_truncated`) — the
successor representation over a single theta sweep — except the intrinsic tuning
width (Fig. 4A–C), which is the asymptotic `c/√λ` scale.

## Running

```bash
pip install -r requirements.txt
python run_all.py                       # regenerates every figure in figures/
python sims/sim2_predictive_inference.py  # …or run one at a time
```

## Simulations

Each script mirrors the experimental setup and plots the result in the style of
the original paper. Figures land in [`figures/`](figures/).

| # | phenomenon | experiment | mechanism in the model |
|---|------------|-----------|------------------------|
| 1 | **Forward theta sweeps & phase precession** | Johnson & Redish (2007); O'Keefe & Recce (1993); Skaggs et al. (1996) | each theta cycle is one **stochastic rollout** (Eq. 5) that sweeps ahead of the animal and serially samples the arms (A–D); a cell fires when the sweep reaches its field, so firing phase precesses (E–F) |
| 2 | **Predictive place-field structure & inferential reasoning** (3 tasks, each with a schematic) | Mehta, Quirk & Wilson (2000); Hollup et al. (2001); Barron et al. (2020, *Cell*) | three readouts of the same predictive map: **A** backward skew — one-way running skews fields backward (place field = resolvent column, learned online, Eq. 21); **B** reward clustering — a goal-directed + dwell policy accumulates the predictive representation at the goal; **C** inferential reasoning — the transitive resolvent reactivates a cue's inferred associate, so RSA of the SR gives associated & **inferred** ≫ non-associated (mouse hippocampal RSA) |
| 3 | **Multi-scale place fields & hyperbolic discounting** | CA1 time constants (doi:10.1002/hipo.70105); tuning widths Cohen & Drugowitsch (2025) / Pfeiffer & Foster (2013); Momennejad & Howard (2018); Masset et al. (2025) | one CA1 leak-rate distribution `λ=1/τ` (`τ` exponential), read out two ways: spatially (width `= c/√λ`, so the pushforward predicts the **aggregated empirical width distribution**, A–C) and temporally (its Laplace transform ⇒ approximately hyperbolic discounting `1/(1+μt)`, D–E) |
| 4 | **DG-gated context representation** in CA1 | Coelho et al. (2024) | the DG context vector `c` gates CA3 via `G` (Eqs. 17–18); contexts remap; reactivating a DG ensemble reinstates its CA1 state and biases retrieval; silencing DG collapses the contexts |
| 5 | **Partial vs complete remapping** in morphed environments | Leutgeb et al. (2005); Wills et al. (2005); Sanders et al. (2020) | per Sanders the difference is the **training, not a parameter**: the two context representations grow more **differentiated** with training (`g`), and a morph cue is assigned to them by similarity, so the transition sharpens on its own. small `g` → partial/gradual (matching Leutgeb); large `g` → complete/abrupt (matching Wills). The same network gives both |
| 6 | **Splitter cells** on a continuous-alternation maze | Wood et al. (2000); Frank et al. (2000) | predictive coding makes the shared stem separate by upcoming arm; the trajectory is a DG-gated context, so stem cells keep their field but change rate by trajectory |

## Repository layout

```
hippo/
  model.py          core network: propagator, resolvent, Laplace bank,
                    rollouts, learning rule, DG context gating
  environments.py   linear track, annular watermaze, T-maze, 2D arena, figure-8
  plotting.py       shared matplotlib style
sims/               one script per figure (sim1 … sim6); Figure 2 is
                    sim2_predictive_inference.py, built from the sim2_backward_skew
                    and sim2_reward_clustering components + the inference RSA
figures/            output figures (PDF)
run_all.py          regenerate all figures
FIGURE_CAPTIONS.md  panel-by-panel captions for every figure
SIMULATION_DETAILS.md  environment, policy, parameters and analysis per simulation
```

See [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md) for per-panel captions,
[SIMULATION_DETAILS.md](SIMULATION_DETAILS.md) for the full setup and parameter
choices of each simulation.
