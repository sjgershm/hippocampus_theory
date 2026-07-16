"""
Reward-clustering component of Simulation 2 — accumulation of place fields at the goal.

Experiment: Hollup, Molden, Donnett, Moser & Moser (2001), "Accumulation of
hippocampal place fields at the goal location in an annular watermaze task",
J. Neurosci. 21:1635-1644.
  - Rats search for a hidden platform (goal) on a narrow annular (ring) water
    track. Place fields accumulate at and just before the goal location: the
    density of fields around the ring peaks at the goal.

Model account: the place-field density predicted by the model is the total
discounted future occupancy that each location receives across the population,
i.e. the column sums of the resolvent R = (lam I - W)^{-1}. Under a goal-
directed swimming policy (trajectories converge on the goal, and the animal
dwells there), this predictive representation accumulates at the goal. Under an
undirected (random foraging) policy it is flat. The accumulation grows as
behaviour becomes more goal-directed with training.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import HippocampalModel, BIN_CM, THETA_PERIOD  # noqa: E402
from hippo.environments import annular_track, annular_track_undirected  # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402


def representation_density(W, lam):
    """Predicted place-field density at each location = column sums of the
    finite-depth (one theta cycle) resolvent (total discounted future occupancy
    received within a sweep), normalized so that 1.0 is the uniform level."""
    R = HippocampalModel(W, lam=lam).resolvent_truncated(THETA_PERIOD)
    d = R.sum(axis=0)
    return d / d.mean()


def run(n=72, goal=0, lam=0.20):
    # trained: goal-directed policy with smooth dwelling near the platform
    W_goal, ang = annular_track(n, goal, drift=1.3, diffusion=0.10,
                                dwell_factor=0.40, dwell_w=6.0)
    # control: undirected random foraging (no goal)
    W_ctrl, _ = annular_track_undirected(n)

    dens_goal = representation_density(W_goal, lam)
    dens_ctrl = representation_density(W_ctrl, lam)

    # signed arc-length distance from goal (cm); ring circumference = n * BIN_CM
    sdist = ((np.arange(n) - goal + n // 2) % n) - n // 2
    arc = sdist * BIN_CM

    # development: interpolate behaviour from undirected -> goal-directed + dwell
    train = np.linspace(0, 1, 12)
    goal_overrep = []
    for a in train:
        Wt, _ = annular_track(n, goal, drift=1.3 * a, diffusion=0.10 + 0.9 * (1 - a),
                              dwell_factor=1.0 - 0.60 * a, dwell_w=6.0)
        dt = representation_density(Wt, lam)
        goal_overrep.append(dt[goal])

    return dict(n=n, goal=goal, ang=ang, arc=arc, dens_goal=dens_goal,
                dens_ctrl=dens_ctrl, train=train,
                goal_overrep=np.array(goal_overrep))


def make_figure(res):
    n, arc = res["n"], res["arc"]
    order = np.argsort(arc)
    fig = plt.figure(figsize=(11, 8))

    # (A) polar schematic of the annular watermaze, colored by field density
    axA = fig.add_subplot(2, 2, 1, projection="polar")
    axA.set_theta_zero_location("N")              # put the goal at the top
    axA.set_theta_direction(-1)
    theta = res["ang"]
    dens = res["dens_goal"]
    width = 2 * np.pi / n
    norm = plt.Normalize(dens.min(), dens.max())
    colors = plt.cm.plasma(norm(dens))
    axA.bar(theta, np.ones(n), width=width, bottom=1.0, color=colors,
            edgecolor="white", linewidth=0.2)
    axA.plot([0], [1.5], marker="*", color="crimson", ms=16,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axA.set_yticklabels([]); axA.set_xticklabels([])
    axA.set_ylim(0, 2.0)
    axA.set_title("A  Annular watermaze", pad=8)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    cb = fig.colorbar(sm, ax=axA, fraction=0.045, pad=0.12)
    cb.set_label("field density (×chance)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # (B) field density around the ring: trained vs control
    axB = fig.add_subplot(2, 2, 2)
    axB.plot(arc[order], res["dens_goal"][order], color="#b30000",
             label="goal-directed (trained)")
    axB.plot(arc[order], res["dens_ctrl"][order], color="0.55",
             label="undirected (control)")
    axB.axhline(1.0, color="0.7", ls="--", lw=1)
    axB.axvline(0, color="crimson", ls=":", lw=1.5)
    axB.set_xlabel("position relative to goal (cm)")
    axB.set_ylabel("place-field density (rel. to chance)")
    axB.set_title("B  Field density")
    axB.legend(fontsize=9)

    # (C) folded by distance from goal (Hollup-style bars)
    axC = fig.add_subplot(2, 2, 3)
    dist = np.abs(((np.arange(n) - res["goal"] + n // 2) % n) - n // 2)
    bins = np.arange(0, n // 2 + 6, 6)
    centers = 0.5 * (bins[:-1] + bins[1:]) * BIN_CM
    dg = np.array([res["dens_goal"][(dist >= bins[i]) & (dist < bins[i + 1])].mean()
                   for i in range(len(bins) - 1)])
    dc = np.array([res["dens_ctrl"][(dist >= bins[i]) & (dist < bins[i + 1])].mean()
                   for i in range(len(bins) - 1)])
    w = (centers[1] - centers[0]) * 0.4
    axC.bar(centers - w / 2, dg, width=w, color="#b30000", label="trained")
    axC.bar(centers + w / 2, dc, width=w, color="0.6", label="control")
    axC.axhline(1.0, color="0.7", ls="--", lw=1)
    axC.set_xlabel("distance from goal (cm)")
    axC.set_ylabel("place-field density (rel. to chance)")
    axC.set_title("C  Distance from goal")
    axC.legend(fontsize=9)

    # (D) accumulation grows with training
    axD = fig.add_subplot(2, 2, 4)
    axD.plot(res["train"] * 100, res["goal_overrep"], "-o", color="#1f6f8b", ms=4)
    axD.axhline(1.0, color="0.7", ls="--", lw=1)
    axD.set_xlabel("goal-directedness of behaviour (% trained)")
    axD.set_ylabel("over-representation at goal")
    axD.set_title("D  Develops with training")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run()
    peak = res["dens_goal"].max()
    print(f"  peak field density at goal = {peak:.2f}x chance "
          f"(control peak = {res['dens_ctrl'].max():.2f}x)")
    print(f"  goal over-representation grows {res['goal_overrep'][0]:.2f}x -> "
          f"{res['goal_overrep'][-1]:.2f}x with training")
    fig = make_figure(res)
    save(fig, "sim2_reward_clustering.png")
