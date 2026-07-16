"""
Simulation 2 — Predictive place-field structure and inferential reasoning.

Three readouts of the same predictive map (the resolvent / successor
representation), each with a task schematic:

  - BACKWARD SKEW (Mehta, Quirk & Wilson 2000): one-way running makes place
    fields skew backward, opposite the direction of travel.
  - REWARD CLUSTERING (Hollup et al. 2001): goal-directed swimming makes the
    predictive representation accumulate at the goal in an annular watermaze.
  - INFERENTIAL REASONING (Barron et al. 2020, Cell, "Neuronal Computation
    Underlying Inferential Reasoning in Humans and Mice"): after learning
    audio->visual and visual->outcome associations, presenting the audio cue
    reactivates the representation of the *inferred* outcome that it was never
    directly paired with. The transitive resolvent binds A->B->C, so a
    representational-similarity analysis (RSA) shows the cue's evoked pattern is
    similar to its associated and its inferred partners, but not to unrelated
    stimuli -- as observed for hippocampal populations in the mouse.

The skew and reward panels reproduce panel A of the former Figures 2 and 3.
"""

import os
import sys

import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hippo import HippocampalModel, project_to_generator, BIN_CM, THETA_PERIOD  # noqa: E402
from hippo.plotting import plt, save, panel_label                               # noqa: E402
from sim2_backward_skew import run as run_skew                                  # noqa: E402
from sim2_reward_clustering import run as run_reward                            # noqa: E402


# --------------------------------------------------------------------------- #
#  inferential reasoning (Barron et al. 2020): associative inference via the SR
# --------------------------------------------------------------------------- #
def run_inference(n=12, lam=0.20, seed=1):
    """Learn audio_i -> visual_i -> outcome_i for n independent triples, then
    measure the representational similarity between the pattern evoked by each
    audio cue and its associated / inferred / unrelated partners.

    Representation of a stimulus = its row of the (one-theta-cycle) successor
    representation R; RSA similarity = correlation between these patterns. The
    audio cue's transitive SR includes its associated visual cue AND its inferred
    outcome, so both are reactivated even though audio->outcome was never paired.
    Per-triple association strengths are heterogeneous (lognormal) to give a
    spread across cues, as in the mouse data (Barron et al. 2020, Fig. 4F).
    """
    rng = np.random.default_rng(seed)
    N = 3 * n
    A = lambda i: i               # audio cues   0 .. n-1
    B = lambda i: n + i           # visual cues  n .. 2n-1
    C = lambda i: 2 * n + i       # outcomes     2n .. 3n-1

    r_av = rng.lognormal(0.0, 0.3, n)             # audio->visual rates
    r_vo = rng.lognormal(0.0, 0.3, n)             # visual->outcome rates
    W = np.zeros((N, N))
    for i in range(n):
        W[A(i), B(i)] = r_av[i]                   # preconditioning: audio -> visual
        W[B(i), C(i)] = r_vo[i]                   # conditioning:    visual -> outcome
    W = project_to_generator(W)
    R = HippocampalModel(W, lam=lam).resolvent_truncated(THETA_PERIOD)

    def rsa(x, y):                # representational similarity = pattern correlation
        a, b = R[x] - R[x].mean(), R[y] - R[y].mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # per audio cue: similarity to its associated visual, its inferred outcome,
    # and the mean over all unrelated visual/outcome representations
    assoc = np.array([rsa(A(i), B(i)) for i in range(n)])
    infer = np.array([rsa(A(i), C(i)) for i in range(n)])
    nonassoc = np.array([np.mean([rsa(A(i), B(j)) for j in range(n) if j != i] +
                                 [rsa(A(i), C(j)) for j in range(n) if j != i])
                         for i in range(n)])
    return dict(n=n, assoc=assoc, infer=infer, nonassoc=nonassoc)


def run():
    return dict(sk=run_skew(), rw=run_reward(), inf=run_inference())


# --------------------------------------------------------------------------- #
#  task schematics
# --------------------------------------------------------------------------- #
def schematic_track(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0.28, 0.86); ax.axis("off")
    ax.add_patch(Rectangle((0.06, 0.40), 0.88, 0.22, facecolor="#ededed",
                           edgecolor="0.6", lw=1))
    ax.annotate("", xy=(0.86, 0.51), xytext=(0.14, 0.51),      # arrow inside the track
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="0.35"))
    ax.text(0.5, 0.73, "one-way running", ha="center", fontsize=8.5)


def schematic_annulus(ax):
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.12, 1.12)   # wide box so the title clears the label
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1.0, 0, 360, width=0.34, facecolor="#dce6f2",
                       edgecolor="0.6", lw=1))
    # optimal policy: from a RANDOMIZED start (dots) take the SHORTER arc to the
    # goal, so trajectories converge on it from both sides (clockwise from one
    # side, counter-clockwise from the other) along the annular track
    r = 0.83                                            # mid-band radius (band r 0.66–1.0)

    def arc_to_goal(deg0, deg1, col="0.4"):
        th = np.radians(np.linspace(deg0, deg1, 80))
        ax.plot(r * np.cos(th), r * np.sin(th), color=col, lw=1.7,
                solid_capstyle="round", zorder=4)
        ax.annotate("", xy=(r * np.cos(th[-1]), r * np.sin(th[-1])),
                    xytext=(r * np.cos(th[-5]), r * np.sin(th[-5])),
                    arrowprops=dict(arrowstyle="-|>", lw=1.7, color=col))
        ax.plot(r * np.cos(th[0]), r * np.sin(th[0]), "o", ms=4.5, color=col,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)  # random start

    arc_to_goal(-30, 66)      # start on the right  → counter-clockwise to goal
    arc_to_goal(210, 114)     # start on the left   → clockwise to goal
    ax.plot(0, r, marker="*", ms=16, color="crimson", markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    ax.text(0, 1.02, "goal", ha="center", fontsize=8.5, color="crimson")


def schematic_inference(ax):
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 0.9); ax.axis("off")
    nodes = [("tone", 0.17, "#9ecae1"), ("image", 0.5, "#a1d99b"),
             ("reward", 0.83, "#fc9272")]
    y = 0.64
    for name, x, col in nodes:
        ax.add_patch(Circle((x, y), 0.10, facecolor=col, edgecolor="0.4", lw=1))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5)
    for x0, x1 in ((0.17, 0.5), (0.5, 0.83)):
        ax.annotate("", xy=(x1 - 0.11, y), xytext=(x0 + 0.11, y),
                    arrowprops=dict(arrowstyle="->", lw=1.7, color="k"))
    ax.text(0.335, y + 0.15, "learn", ha="center", fontsize=7.5)
    ax.text(0.665, y + 0.15, "learn", ha="center", fontsize=7.5)
    ax.annotate("", xy=(0.80, 0.43), xytext=(0.20, 0.43),
                arrowprops=dict(arrowstyle="->", lw=1.7, color="crimson", ls="--",
                                connectionstyle="arc3,rad=0.38"))
    ax.text(0.5, 0.06, "inferred", ha="center", fontsize=8.5, color="crimson")


# --------------------------------------------------------------------------- #
#  figure: schematics (top row) + results (bottom row), one column per task
# --------------------------------------------------------------------------- #
def make_figure(res):
    sk, rw, inf = res["sk"], res["rw"], res["inf"]
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.0], hspace=0.22, wspace=0.36)

    # ---- schematics ----
    axs0 = fig.add_subplot(gs[0, 0]); schematic_track(axs0)
    axs0.set_title("Backward skew", fontsize=11); panel_label(axs0, "A")
    axs1 = fig.add_subplot(gs[0, 1]); schematic_annulus(axs1)
    axs1.set_title("Reward clustering", fontsize=11); panel_label(axs1, "B")
    axs2 = fig.add_subplot(gs[0, 2]); schematic_inference(axs2)
    axs2.set_title("Inferential reasoning", fontsize=11); panel_label(axs2, "C")

    # ---- (A) backward-skewing field across laps ----
    axA = fig.add_subplot(gs[1, 0])
    x, nsk = sk["x"], sk["n"]; j = nsk // 2
    xc = (x - j) * BIN_CM
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(sk["snap_laps"])))
    for c, lap in zip(cmap, sk["snap_laps"]):
        f = sk["snapshots"][lap]
        axA.plot(xc, f / f.max(), color=c, label=f"lap {lap}")
    axA.axvline(0, color="0.6", ls="--", lw=1)
    axA.annotate("", xy=(44, 1.08), xytext=(8, 1.08),
                 arrowprops=dict(arrowstyle="->", color="k"), annotation_clip=False)
    axA.text(26, 1.12, "travel", ha="center", va="bottom", fontsize=8.5, clip_on=False)
    axA.set_ylim(top=1.2)
    axA.set_xlabel("position relative to field peak (cm)")
    axA.set_ylabel("normalized firing rate")
    axA.legend(fontsize=8, loc="upper left")

    # ---- (B) predictive field density around the ring (polar) ----
    axB = fig.add_subplot(gs[1, 1], projection="polar")
    axB.set_theta_zero_location("N"); axB.set_theta_direction(-1)
    nrw = rw["n"]; dens = rw["dens_goal"]
    norm = plt.Normalize(dens.min(), dens.max())
    axB.bar(rw["ang"], np.ones(nrw), width=2 * np.pi / nrw, bottom=1.0,
            color=plt.cm.plasma(norm(dens)), edgecolor="white", linewidth=0.2)
    axB.plot([0], [1.5], marker="*", color="crimson", ms=15,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axB.set_yticklabels([]); axB.set_xticklabels([]); axB.set_ylim(0, 2.0)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    cb = fig.colorbar(sm, ax=axB, orientation="horizontal", fraction=0.05, pad=0.02)
    cb.set_label("density (×chance)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ---- (C) RSA: representational similarity by condition (cf. Barron Fig. 4F) ----
    axC = fig.add_subplot(gs[1, 2])
    jit = np.random.default_rng(0)
    groups = [("associated", inf["assoc"], "#a1d99b"),   # = 'image' oval in the schematic
              ("inferred", inf["infer"], "#fc9272"),      # = 'reward' oval in the schematic
              ("non-assoc.", inf["nonassoc"], "0.6")]
    for k, (lab, v, col) in enumerate(groups):
        axC.bar(k, v.mean(), width=0.64, color=col, edgecolor="k", linewidth=0.5,
                yerr=v.std(ddof=1) / np.sqrt(len(v)), capsize=3, zorder=2)
        axC.plot(k + jit.uniform(-0.14, 0.14, len(v)), v, "o", ms=3, mfc="white",
                 mec="0.3", mew=0.6, zorder=3)
    axC.axhline(0, color="0.5", lw=0.8)
    axC.set_xticks(range(3)); axC.set_xticklabels([g[0] for g in groups], fontsize=9)
    axC.set_ylabel("representational similarity")
    axC.set_ylim(-0.3, 1.05)
    axC.set_title("Inferred reactivation", fontsize=11)

    return fig


if __name__ == "__main__":
    res = run()
    inf = res["inf"]
    print(f"  skew: skewness -> {res['sk']['skew'][-1]:+.2f}")
    print(f"  reward: goal density {res['rw']['dens_goal'].max():.2f}x chance")
    print(f"  inference RSA: associated {inf['assoc'].mean():.2f}, "
          f"inferred {inf['infer'].mean():.2f}, non-associated {inf['nonassoc'].mean():.2f}")
    fig = make_figure(res)
    save(fig, "sim2_predictive_inference.png")
