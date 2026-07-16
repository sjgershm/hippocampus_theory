"""
Simulation 4 — DG gates context-dependent neural states and memory retrieval.

Experiment: Coelho et al. (2024), "Dentate gyrus ensembles gate context-
dependent neural states and memory retrieval", Science Advances 10:eadn9815.
Three panels mirror the paper's key findings:

  A  Context-specific neural states  (their Fig. 3H-J): with distinct DG context
     ensembles the CA1 population settles into two distinct states (high within-
     context, low between-context population-vector correlation).
  B  DG reactivation reinstates the state  (their Fig. 5D-E): given an ambiguous
     cue, driving a context's DG ensemble makes the CA1 state resemble that
     context. The balanced midpoint (no DG bias) is the ambiguous / DG-silenced
     condition of their Fig. 1.
  C  Context-appropriate memory retrieval  (their Fig. 2D-E, F-G, the headline
     result): the reinstated state biases retrieval toward that context's
     learned association; with no DG bias there is no preference (their Fig.
     1G-H, where silencing DG made mice dig equally in both wells).

Model account (Eqs. 17-18): the DG context vector c gates CA3 recurrent dynamics
through G, giving an effective generator J_ij = W_ij b_j with b = c G. Different
contexts route CA3 to different effective generators, so the CA1 readout
y = x (lam I - J)^{-1} remaps between contexts; reinstating a context's DG input
reinstates that context's CA1 state and its association.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import ContextHippocampalModel, THETA_PERIOD  # noqa: E402
from hippo.environments import annular_track_undirected  # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402


def context_gates(n, rng, frac=0.6, lo=0.05):
    """Two sparse, partially-overlapping DG context ensembles. A subset of CA3
    columns is gated ON (gate 1) per context, the rest nearly OFF (gate `lo`).
    The two ensembles are anti-correlated (overlap = 2*frac - 1), so distinct
    contexts engage distinct ensembles -> remapping, while a shared core keeps
    remapping partial (cf. partial remapping)."""
    u = rng.random(n)
    g_A = np.where(u < frac, 1.0, lo)
    g_B = np.where(u > 1 - frac, 1.0, lo)
    return np.vstack([g_A, g_B])


def bump_inputs(n, sigma=1.6):
    x = np.arange(n)
    X = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    return X / X.sum(1, keepdims=True)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run(n=30, lam=0.20, seed=3):
    rng = np.random.default_rng(seed)
    W, _ = annular_track_undirected(n, diffusion=1.0)     # shared spatial graph
    # two context-specific DG gating patterns (rows of G)
    G = context_gates(n, rng)
    model = ContextHippocampalModel(W, G, lam=lam)

    c_A, c_B = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    X = bump_inputs(n)
    T = THETA_PERIOD                                      # one-theta-cycle SR

    # (A) context-specific neural states: population-vector correlation matrix
    #     [A;B] x [A;B] under the two intact DG contexts
    YA = np.array([model.output_ctx(X[i], c_A, T=T) for i in range(n)])
    YB = np.array([model.output_ctx(X[i], c_B, T=T) for i in range(n)])
    PV = np.corrcoef(np.vstack([YA, YB]))
    within = np.mean([corr(YA[i], YA[(i + 1) % n]) for i in range(n)])   # same context
    between = np.mean([corr(YA[i], YB[i]) for i in range(n)])            # across contexts

    # (B, C) drive the DG ensemble from context A (alpha=0) to context B (alpha=1)
    #        given an AMBIGUOUS cue; alpha=0.5 is the balanced / DG-silenced case.
    x_amb = np.ones(n) / n
    alphas = np.linspace(0, 1, 21)
    tA = model.output_ctx(x_amb, c_A, T=T)                # pure context-A state
    tB = model.output_ctx(x_amb, c_B, T=T)                # pure context-B state
    sim_A, sim_B = [], []
    for a in alphas:
        y = model.output_ctx(x_amb, (1 - a) * c_A + a * c_B, T=T)
        sim_A.append(corr(y, tA)); sim_B.append(corr(y, tB))
    sim_A, sim_B = np.array(sim_A), np.array(sim_B)

    # retrieval bias: P(retrieve context-A association) from the reinstated state
    beta = 3.5
    pA = np.exp(beta * sim_A) / (np.exp(beta * sim_A) + np.exp(beta * sim_B))

    return dict(n=n, PV=PV, within=within, between=between,
                alphas=alphas, sim_A=sim_A, sim_B=sim_B, pA=pA)


def make_figure(res):
    n = res["n"]
    a = res["alphas"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (A) context-specific neural states: PV correlation matrix (intact DG)  [Fig 3H-J]
    im = ax[0].imshow(res["PV"], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax[0].axhline(n - 0.5, color="k", lw=1); ax[0].axvline(n - 0.5, color="k", lw=1)
    ax[0].set_xticks([n / 2, n + n / 2]); ax[0].set_xticklabels(["ctx A", "ctx B"])
    ax[0].set_yticks([n / 2, n + n / 2])
    ax[0].set_yticklabels(["ctx A", "ctx B"], rotation=90, va="center")
    ax[0].set_title("A  Context-specific neural states")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label("CA1 pop-vec correlation", fontsize=8); cb.ax.tick_params(labelsize=7)
    _bb = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2)
    ax[0].text(n / 2, n / 2, "within", ha="center", va="center", fontsize=8, bbox=_bb)
    ax[0].text(n + n / 2, n / 2, "across", ha="center", va="center", fontsize=8, bbox=_bb)
    ax[0].text(n / 2, n + n / 2, "across", ha="center", va="center", fontsize=8, bbox=_bb)
    ax[0].text(n + n / 2, n + n / 2, "within", ha="center", va="center", fontsize=8, bbox=_bb)

    # (B) DG reactivation reinstates the context-specific state  [Fig 5D-E]
    ax[1].plot(a, res["sim_A"], "-o", color="#2166ac", ms=3, label="similarity to context A")
    ax[1].plot(a, res["sim_B"], "-o", color="#b2182b", ms=3, label="similarity to context B")
    ax[1].axvline(0.5, color="0.6", ls=":", lw=1)
    ax[1].set_xticks([0, 0.5, 1]); ax[1].set_xticklabels(["A", "balanced", "B"])
    ax[1].set_xlabel("DG ensemble driven")
    ax[1].set_ylabel("CA1 state similarity")
    ax[1].set_title("B  DG reactivation reinstates the state")
    ax[1].legend(fontsize=8, loc="center right")

    # (C) context-appropriate memory retrieval  [Fig 2D-E, F-G; balanced ~ Fig 1G-H]
    ax[2].plot(a, res["pA"], color="#6a51a3", lw=2.5)
    ax[2].axhline(0.5, color="0.6", ls="--", lw=1)
    ax[2].axvline(0.5, color="0.6", ls=":", lw=1)
    ax[2].fill_between(a, 0.5, res["pA"], where=res["pA"] >= 0.5, color="#2166ac", alpha=0.15)
    ax[2].fill_between(a, res["pA"], 0.5, where=res["pA"] < 0.5, color="#b2182b", alpha=0.15)
    ax[2].set_xticks([0, 0.5, 1]); ax[2].set_xticklabels(["A", "balanced", "B"])
    ax[2].set_xlabel("DG ensemble driven")
    ax[2].set_ylabel("P(retrieve context-A association)")
    ax[2].set_title("C  Context-appropriate retrieval")
    ax[2].set_ylim(-0.02, 1.02)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run()
    mid = len(res["alphas"]) // 2
    print(f"  context separation: within={res['within']:.2f}, between={res['between']:.2f}"
          f"  (Δ = {res['within'] - res['between']:.2f})")
    print(f"  retrieval P(A): reactivate-A={res['pA'][0]:.2f}, balanced={res['pA'][mid]:.2f}, "
          f"reactivate-B={res['pA'][-1]:.2f}")
    fig = make_figure(res)
    save(fig, "sim4_context_dg.png")
