"""
Figure 5 panel D — experienced morph frequency shapes remapping.

Plitt & Giocomo (2021, *Nature Neuroscience* 24:705-714, "Experience-dependent
contextual codes in the hippocampus") morphed a VR environment between two
endpoints and manipulated HOW OFTEN intermediate morphs were experienced:

    Rare Morph      (4 mice)  <4% of trials at intermediate morphs (0.25-0.75)
    Frequent Morph  (5 mice)  60% of trials at intermediate morphs

Morph values were 0, 0.25, 0.5, 0.75, 1 with +/-0.1 jitter. Note that "60% over
three intermediate values" is 20% each, so the Frequent group's experience is
*uniform* over the five morphs -- the flat prior of their model.

Result (their Fig. 2, 3f-g): Rare-Morph mice held one representation up to a
threshold morph and then switched coherently (step-like; bimodal Similarity
Fraction), while Frequent-Morph mice changed gradually (unimodal SF);
KS D = 0.395. They account for this with a MAP estimate under a learned prior.

Model. This is the same DG-gated network as Simulation 5, with the context
vector supplied by inference over a latent context s given the morph cue m:

    prior      pi(s)   = experienced morph frequencies, smoothed by the jitter
    likelihood p(m|s)  = N(m; s, sigma_obs^2)
    context    c(m)    = [1 - E[s|m], E[s|m]]

Everything else -- the ring, the overlapping DG ensembles, the truncated
resolvent read-out -- is Simulation 5 unchanged. The *only* thing that differs
between the two groups is the prior, i.e. what they experienced.

This module is a COMPONENT of Figure 5: `sim5_partial_remapping.py` imports
`run` for panel D. Run it directly to print the numbers, including the
Similarity-Fraction / KS comparison that the combined figure does not plot.

Why the manipulation works: with a FLAT prior the posterior mean tracks the cue
veridically, E[s|m] ~ m, whatever the sensory noise. With a BIMODAL prior an
ambiguous cue is resolved by falling back on the prior, which says "it is one of
the two endpoints" -- so the representation switches categorically. Sensory
uncertainty is therefore what converts a bimodal prior into a step-like code,
and it leaves the flat-prior group almost untouched (see `report()`).
"""

import os
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import ContextHippocampalModel, THETA_PERIOD      # noqa: E402
from hippo.environments import annular_track_undirected      # noqa: E402
from hippo import context_gates, bump_inputs, flat_corr  # noqa: E402

MORPHS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
W_RARE = np.array([0.48, 0.04 / 3, 0.04 / 3, 0.04 / 3, 0.48])   # <4% intermediate
W_FREQ = np.full(5, 0.20)                                       # 60% intermediate
SIGMA_JIT = 0.10        # the +/-0.1 per-feature jitter they applied
SIGMA_OBS = 0.20        # the animal's uncertainty about the morph cue
C_RARE, C_FREQ = "#b2182b", "#2166ac"
GROUPS = (("Rare Morph", W_RARE, C_RARE), ("Frequent Morph", W_FREQ, C_FREQ))


# --------------------------------------------------------------------------- #
#  inference over the latent context
# --------------------------------------------------------------------------- #
def latent_grid(n=901):
    return np.linspace(-0.4, 1.4, n)


def learned_prior(w, s=None, sigma_jit=SIGMA_JIT):
    """What the animal experienced: the morph frequencies smeared by the jitter."""
    s = latent_grid() if s is None else s
    p = sum(wi * np.exp(-0.5 * ((s - v) / sigma_jit) ** 2)
            for wi, v in zip(w, MORPHS))
    return p / p.sum()


def posterior_mean(m, prior, s=None, sigma_obs=SIGMA_OBS):
    """E[s | m]: the inferred context, used as the DG context vector."""
    s = latent_grid() if s is None else s
    post = prior * np.exp(-0.5 * ((np.atleast_1d(m)[:, None] - s) / sigma_obs) ** 2)
    return (post * s).sum(1) / post.sum(1)


# --------------------------------------------------------------------------- #
def run(n=30, lam=0.20, seed=2, n_trials=4000, sigma_obs=SIGMA_OBS):
    rng = np.random.default_rng(seed)
    W, _ = annular_track_undirected(n, diffusion=1.0)
    model = ContextHippocampalModel(W, context_gates(n, rng), lam=lam)
    X = bump_inputs(n)
    s = latent_grid()

    def ca1(shat):
        c = np.array([1.0 - shat, shat])
        return np.array([model.output_ctx(X[i], c, T=THETA_PERIOD) for i in range(n)])

    end0, end1 = ca1(0.0), ca1(1.0)

    def _raw_sf(shat):
        m_ = ca1(shat)
        r0 = max(flat_corr(m_, end0), 0.0)
        r1 = max(flat_corr(m_, end1), 0.0)
        return r0 / (r0 + r1 + 1e-12)

    # The two DG ensembles overlap (shared core), so even the pure context-1 map
    # stays correlated with the context-0 map and the raw ratio spans only a
    # sub-interval. Rescale so the two endpoints read 0 and 1, which is what
    # makes the Similarity Fraction comparable to the reported one.
    _sf1, _sf0 = _raw_sf(1.0), _raw_sf(0.0)

    def sim_fraction(shat):
        """Similarity Fraction: 1 at the 0 end, 0 at the 1 end."""
        span = _sf0 - _sf1
        return (_raw_sf(shat) - _sf1) / span if abs(span) > 1e-9 else 0.5

    morph = np.linspace(0, 1, 41)
    out = {}
    for name, w, col in GROUPS:
        pri = learned_prior(w, s)
        shat = posterior_mean(morph, pri, s, sigma_obs)
        sim0 = np.array([flat_corr(ca1(v), end0) for v in shat])
        # trials: a morph value from the test set (all five, equally) + jitter
        if n_trials:
            v = rng.choice(MORPHS, n_trials) + rng.normal(0, SIGMA_JIT, n_trials)
            sf = np.array([sim_fraction(x) for x in
                           posterior_mean(v, pri, s, sigma_obs)])
        else:
            sf = np.empty(0)
        out[name] = dict(prior=pri, shat=shat, sim0=sim0, sf=sf, color=col,
                         slope=float((posterior_mean(np.array([0.55]), pri, s, sigma_obs)[0]
                                      - posterior_mean(np.array([0.45]), pri, s, sigma_obs)[0]) / 0.1))
    ks = (ks_2samp(out["Rare Morph"]["sf"], out["Frequent Morph"]["sf"])
          if n_trials else None)
    return dict(s=s, morph=morph, out=out, ks=ks, sigma_obs=sigma_obs)


def report(res):
    print(f"  sigma_obs = {res['sigma_obs']}, sigma_jit = {SIGMA_JIT}")
    for name in ("Rare Morph", "Frequent Morph"):
        d = res["out"][name]
        sf = d["sf"]
        mid = float(np.mean((sf > 0.25) & (sf < 0.75)))
        print(f"  {name:<15} transition slope at m=0.5: {d['slope']:.2f}   "
              f"SF in the middle band: {mid:.0%}")
    print(f"  KS between the two SF distributions: D = {res['ks'].statistic:.3f}, "
          f"p = {res['ks'].pvalue:.1e}   (reported: D = 0.395)")
    print("\n  sensitivity to the one free parameter, sigma_obs:")
    print(f"    {'sigma_obs':>10} {'rare slope':>11} {'freq slope':>11} {'KS D':>8}")
    for so in (0.10, 0.20, 0.30, 0.40):
        r = run(n_trials=1500, sigma_obs=so)
        print(f"    {so:>10.2f} {r['out']['Rare Morph']['slope']:>11.2f} "
              f"{r['out']['Frequent Morph']['slope']:>11.2f} {r['ks'].statistic:>8.3f}")
    print("    (the step-vs-gradual dissociation is robust across sigma_obs; D peaks")
    print("     near 0.28, below the reported 0.395 -- see the caption)")


if __name__ == "__main__":
    # Component of Figure 5 (panel D); running it directly just reports the
    # numbers, including the Similarity-Fraction comparison that the combined
    # figure no longer plots.
    report(run())
