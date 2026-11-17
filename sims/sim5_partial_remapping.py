"""
Simulation 5 — Partial vs complete remapping in morphed environments.

Phenomena:
  - Leutgeb et al. (2005): morphing an environment (square -> circle) produced
    *partial*, gradual remapping between the extremes.
  - Wills et al. (2005): a similar morph produced *complete*, coherent remapping
    with a sharp transition.

Sanders, Wilson & Gershman (2020) argue the two results differ because of the
training DATA and protocol (the studies were not a controlled early-vs-late
comparison), not a model parameter. We capture this with a single, minimal
assumption and NO added machinery: the two stored context representations become
increasingly **differentiated** with training. A morph cue is assigned to the two
contexts by similarity, so as the representations separate the assignment — and
hence the remapping — sharpens on its own:

    s(m) = logistic( g * (2m - 1) ),      c(m) = [ 1 - s(m),  s(m) ],

where m in [0,1] is the morph level (square -> circle) and g >= 0 is the context
differentiation, which grows monotonically with training. A single g controls
BOTH signatures of the transition:
    endpoint separation  |s(1) - s(0)| = tanh(g/2)   -> degree of remapping,
    midpoint slope        s'(1/2)      = g/2          -> abruptness,
both increasing with g. c(m) gates CA3 (Eqs. 17-18); the two DG ensembles overlap
(shared core, `frac`), so remapping is partial at the cell level even when
complete at the population level.

    g small (low  differentiation)  -> s ~ 0.5, flat   -> gradual, partial  (Leutgeb)
    g large (high differentiation)  -> s steps 0 -> 1  -> abrupt,  complete (Wills)

The same network gives both; only the differentiation g differs (which we assume
grows with training). Note the two studies did not manipulate training in a
controlled way, so we do not label the regimes as "early"/"late" training — g is
a model state that matches Leutgeb's partial and Wills's complete remapping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import ContextHippocampalModel, THETA_PERIOD  # noqa: E402
from hippo import context_gates, bump_inputs, flat_corr  # noqa: E402
from hippo.environments import annular_track_undirected  # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402
from sim5b_morph_frequency import run as run_morph_frequency  # noqa: E402

DIFF_LO = 1.0    # low  differentiation  -> matches Leutgeb (partial)
DIFF_HI = 6.0    # high differentiation  -> matches Wills   (complete)


def context_vector(m, g):
    """DG context for morph level m at differentiation g (assumed to grow with
    training). Similarity-based soft assignment to the two (increasingly
    separated) contexts."""
    s = 1.0 / (1.0 + np.exp(-g * (2 * m - 1)))
    return np.array([1 - s, s])


def run(n=30, lam=0.20, seed=2, diff_lo=DIFF_LO, diff_hi=DIFF_HI):
    rng = np.random.default_rng(seed)
    W, _ = annular_track_undirected(n, diffusion=1.0)
    model = ContextHippocampalModel(W, context_gates(n, rng), lam=lam)
    X = bump_inputs(n)

    def ca1_maps(m, g):
        c = context_vector(m, g)
        return np.array([model.output_ctx(X[i], c, T=THETA_PERIOD) for i in range(n)])

    # (A) morph response: CA1 similarity to the square map across the morph
    morph = np.linspace(0, 1, 21)
    out = {}
    for name, g in (("lo", diff_lo), ("hi", diff_hi)):
        maps = [ca1_maps(m, g) for m in morph]
        sq = maps[0]
        out[name] = np.array([flat_corr(mm, sq) for mm in maps])

    # (B) remapping between the two extremes vs differentiation
    diffs = np.linspace(0.0, diff_hi * 1.1, 24)
    remap = np.array([1 - flat_corr(ca1_maps(0.0, g), ca1_maps(1.0, g)) for g in diffs])

    # (C) per-cell remapping across the morph at high differentiation
    maps_hi = [ca1_maps(m, diff_hi) for m in morph]
    act = np.array([mm.mean(0) for mm in maps_hi])          # morph x cell
    order = np.argsort(act[-1] - act[0])
    act_norm = act[:, order] / (act[:, order].max(0, keepdims=True) + 1e-12)
    change = np.abs(act[-1] - act[0]) / (np.maximum(act[0], act[-1]) + 1e-12)
    frac_remap = float(np.mean(change > 0.5))

    # panel D: the same contrast produced by a *manipulated* variable --
    # how often intermediate morphs were experienced (Plitt & Giocomo 2021)
    mf = run_morph_frequency(n=n, lam=lam, seed=seed, n_trials=0)

    return dict(n=n, morph=morph, out=out, diffs=diffs, remap=remap,
                diff_lo=diff_lo, diff_hi=diff_hi,
                act_norm=act_norm, frac_remap=frac_remap, mf=mf)


def make_figure(res):
    # 2 x 2. The panels are wider than in the original 1x3 layout, so the
    # centred titles clear the bold panel labels.
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    axA, axB = ax[0]
    axC, axD = ax[1]
    mp = res["morph"] * 100

    # (A) morph response: gradual (low g, Leutgeb) vs abrupt (high g, Wills)
    axA.plot(mp, res["out"]["lo"], "-o", color="#2166ac", ms=3,
             label="low differentiation\n— partial (Leutgeb)")
    axA.plot(mp, res["out"]["hi"], "-o", color="#b2182b", ms=3,
             label="high differentiation\n— complete (Wills)")
    axA.set_xlabel("morph level (% toward circle)")
    axA.set_ylabel("CA1 similarity to square")
    axA.set_title("A  Gradual vs abrupt remapping")
    axA.legend(fontsize=7.5, loc="lower left")

    # (B) remapping between extremes grows with context differentiation
    axB.plot(res["diffs"], res["remap"], "-o", color="#1b7837", ms=3)
    axB.set_xlabel("context differentiation")
    axB.set_ylabel("remapping (1 − square/circle corr)")
    axB.set_title("B  Remapping vs differentiation")

    # (C) per-cell partial remapping (high g): some cells switch, a shared core
    #     does not
    im = axC.imshow(res["act_norm"].T, cmap="magma", origin="lower",
                    aspect="auto", extent=[0, 100, 0, res["n"]])
    axC.set_xlabel("morph level (% toward circle)")
    axC.set_ylabel("cell (sorted)")
    axC.set_title("C  Partial remapping (per cell)")
    fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="norm. activity")

    # (D) the same contrast, driven by how often intermediate morphs were
    #     experienced rather than by the differentiation parameter
    mf = res["mf"]
    for name in ("Rare Morph", "Frequent Morph"):
        d = mf["out"][name]
        axD.plot(mf["morph"] * 100, d["sim0"], "-o", color=d["color"], ms=3,
                 label=name)
    axD.set_xlabel("morph level (% toward circle)")
    axD.set_ylabel("CA1 similarity to the 0 end")
    axD.set_title("D  Experienced morph frequency")
    axD.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run()
    print(f"  differentiation: low g={res['diff_lo']}, high g={res['diff_hi']}")
    print(f"  morph similarity to square at circle end: low g={res['out']['lo'][-1]:.2f} "
          f"(partial), high g={res['out']['hi'][-1]:.2f} (complete)")
    print(f"  remapping between extremes: low g={res['remap'][np.argmin(np.abs(res['diffs']-res['diff_lo']))]:.2f}, "
          f"high g={res['remap'][np.argmin(np.abs(res['diffs']-res['diff_hi']))]:.2f}")
    print(f"  high-differentiation per-cell remapping: {res['frac_remap']:.0%} of cells change")
    fig = make_figure(res)
    save(fig, "sim5_partial_remapping.png")
