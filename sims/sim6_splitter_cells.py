"""
Simulation 6 — Splitter cells on a continuous alternation maze.

Phenomenon: Frank et al. (2000); Wood et al. (2000). On a figure-8 / W-maze
continuous-alternation task, many place cells with fields on the *shared*
central stem fire at different rates depending on the animal's trajectory
(which way it just came / is about to go), even though location and behavior on
the stem are identical -- "splitter" cells.

Model account: the hippocampal code is predictive AND context-dependent. (i)
Prospectively, the resolvent on the stem already represents the upcoming arm
(successor states), so the population near the choice point separates by future
path. (ii) The trajectory itself is a latent context maintained by the DG,
which gates CA3 (Eqs. 17-18); the two trajectory contexts engage partly
different gains, so a stem cell keeps its field location but changes firing
RATE with trajectory -- rate remapping = splitting. Both effects fall out of
the same circuit used for prediction and context.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import ContextHippocampalModel, BIN_CM, THETA_PERIOD  # noqa: E402
from hippo.environments import figure_eight              # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402


def traj_gates(n, rng, left, right, amp=0.8, cutoff=0.12, lo=0.05):
    """Two anti-correlated, smooth per-cell DG gains (one per trajectory) plus
    routing of the correct return loop. Where the gains differ a stem cell
    splits; where they agree (shared core) it does not — partial splitting. The
    smooth field takes both signs across the stem, so some stem cells prefer
    left trials and others prefer right."""
    f = rng.normal(size=n)
    k = np.fft.rfftfreq(n)
    h = np.fft.irfft(np.fft.rfft(f) * np.exp(-(k / cutoff) ** 2), n)
    h = h / (np.abs(h).max() + 1e-9)               # in [-1, 1]
    gL = np.clip(1 + amp * h, 0.1, None)
    gR = np.clip(1 - amp * h, 0.1, None)
    gL[right] = lo; gR[left] = lo                   # route the correct loop
    return gL, gR


def run(stem=12, loop=16, lam=0.20, lo=0.05, seed=4):
    rng = np.random.default_rng(seed)
    env = figure_eight(stem=stem, loop=loop)
    W, coords = env["W"], env["coords"]
    stem_idx, choice = env["stem"], env["choice"]
    left, right, n = env["left"], env["right"], env["n"]

    # two trajectory contexts: anti-correlated DG gains + route the return loop
    gL, gR = traj_gates(n, rng, left, right, lo=lo)
    model = ContextHippocampalModel(W, np.vstack([gL, gR]), lam=lam)
    RL = model.resolvent_ctx([1, 0], T=THETA_PERIOD)      # one-theta-cycle SR
    RR = model.resolvent_ctx([0, 1], T=THETA_PERIOD)

    # firing-rate maps on the stem (cell c at stem position p -> R[p, c])
    fireL = RL[stem_idx, :]
    fireR = RR[stem_idx, :]

    # which cells have their field on the stem
    peak = np.array([np.argmax(np.maximum(RL[:, c], RR[:, c])) for c in range(n)])
    stem_cells = np.where(np.isin(peak, stem_idx))[0]
    stem_cells = stem_cells[np.argsort(peak[stem_cells])]

    # in-field firing rate (at each cell's peak stem position), L vs R trials
    pk = peak[stem_cells]
    actL = RL[pk, stem_cells]
    actR = RR[pk, stem_cells]
    split = np.abs(actL - actR) / (np.maximum(actL, actR) + 1e-9)
    frac_split = float(np.mean(split > 0.2))

    # trajectory selectivity along the stem: 1 - PV correlation(L, R)
    sel = []
    for p in range(len(stem_idx)):
        a, b = fireL[p] - fireL[p].mean(), fireR[p] - fireR[p].mean()
        sel.append(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    sel = np.array(sel)

    # example splitter cells with localized mid-stem fields; include at least one
    # of each preference (left-trial rate > right, and right > left)
    lo_b, hi_b = stem_idx[0] + 2, stem_idx[-1] - 1
    mid = np.array([c for c in stem_cells if lo_b <= peak[c] <= hi_b])
    dL, dR = RL[peak[mid], mid], RR[peak[mid], mid]
    signed = (dL - dR) / (np.maximum(dL, dR) + 1e-9)   # >0: left-preferring, <0: right
    left_best = mid[int(np.argmax(signed))]            # strongest left-preferring
    right_best = mid[int(np.argmin(signed))]           # strongest right-preferring
    rank = mid[np.argsort(-np.abs(signed))]
    third = int(next(c for c in rank if c not in (left_best, right_best)))
    examples = np.array([left_best, right_best, third])
    examples = examples[np.argsort(peak[examples])]    # order along the stem

    return dict(env=env, coords=coords, stem_idx=stem_idx, choice=choice,
                left=left, right=right, RL=RL, RR=RR, fireL=fireL, fireR=fireR,
                stem_cells=stem_cells, peak=peak, actL=actL, actR=actR,
                split=split, frac_split=frac_split, sel=sel, examples=examples)


def make_figure(res):
    coords = res["coords"]
    env = res["env"]
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # (A) maze schematic
    axA = fig.add_subplot(gs[0, 0])
    W = env["W"]
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[0]):
            if W[i, j] > 1e-9 or W[j, i] > 1e-9:
                axA.plot(coords[[i, j], 0], coords[[i, j], 1], "-",
                         color="0.85", lw=5, solid_capstyle="round", zorder=1)
    axA.plot(coords[res["left"], 0], coords[res["left"], 1], ".", color="#2166ac", ms=8)
    axA.plot(coords[res["right"], 0], coords[res["right"], 1], ".", color="#b2182b", ms=8)
    axA.plot(coords[res["stem_idx"], 0], coords[res["stem_idx"], 1], "-",
             color="k", lw=6, solid_capstyle="round", zorder=3)
    axA.plot(*coords[res["choice"]], "^k", ms=11, zorder=4)
    axA.text(0.7, coords[res["stem_idx"]][len(res["stem_idx"]) // 2, 1],
             "shared\nstem", fontsize=9, fontweight="bold", ha="left")
    axA.text(coords[res["left"], 0].min() - 0.3, 0.3, "left\nloop", color="#2166ac",
             fontsize=9, ha="center")
    axA.text(coords[res["right"], 0].max() + 0.3, 0.3, "right\nloop", color="#b2182b",
             fontsize=9, ha="center")
    axA.set_aspect("equal"); axA.axis("off")
    axA.set_title("A  Alternation maze")

    # (B) example splitter cells: rate vs stem position, L vs R (offset stack)
    axB = fig.add_subplot(gs[0, 1:])
    sp = np.arange(len(res["stem_idx"])) * BIN_CM
    step = 1.35
    for k, c in enumerate(res["examples"]):
        scale = max(res["fireL"][:, c].max(), res["fireR"][:, c].max()) + 1e-9
        off = k * step
        axB.plot(sp, res["fireL"][:, c] / scale + off, "-o", color="#2166ac", ms=3,
                 label="left trial" if k == 0 else None)
        axB.plot(sp, res["fireR"][:, c] / scale + off, "-o", color="#b2182b", ms=3,
                 label="right trial" if k == 0 else None)
        axB.axhline(off, color="0.9", lw=0.8)
        axB.text(0, off + 0.55, f"stem cell {c}", fontsize=8, color="0.3")
    axB.set_ylim(-0.25, (len(res["examples"]) - 1) * step + 1.6)   # headroom for legend
    axB.set_xlabel("position along shared stem, cm (start → choice point)")
    axB.set_ylabel("normalized firing rate (offset per cell)")
    axB.set_yticks([])
    axB.set_title("B  Splitter cells")
    axB.legend(fontsize=9, loc="upper right")

    return fig


if __name__ == "__main__":
    res = run()
    print(f"  {len(res['stem_cells'])} stem cells; mean split index = "
          f"{res['split'].mean():.2f}; {res['frac_split']:.0%} split (>0.2)")
    print(f"  trajectory selectivity along stem: {res['sel'][0]:.2f} (start) -> "
          f"{res['sel'][-1]:.2f} (choice)")
    fig = make_figure(res)
    save(fig, "sim6_splitter_cells.png")
