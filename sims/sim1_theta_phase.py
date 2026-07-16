"""
Simulation 1 — Forward theta sweeps and phase precession.

Both phenomena are the same mechanism (Eq. 5): each theta cycle carries one
forward rollout of the generator from the animal's current state. The decoded
position sweeps ahead of the animal within the cycle.

  - Johnson & Redish (2007): at a choice point the rollout sweeps forward and
    serially samples the arms across cycles (panels A-D).
  - O'Keefe & Recce (1993); Skaggs et al. (1996): a place cell fires when the
    sweep reaches its field, so as the animal approaches, the firing phase
    advances -- phase precession (panels E-F).
"""

import os
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import HippocampalModel, BIN_CM, THETA_PERIOD  # noqa: E402
from hippo.environments import t_maze, linear_track      # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402


def graph_distance(W):
    A = (W - np.diag(np.diag(W)) > 1e-9).astype(float)
    A = np.maximum(A, A.T)
    return shortest_path(A, method="D", unweighted=True)


# --------------------------------------------------------------------------- #
#  forward sweeps (T-maze + linear track)
# --------------------------------------------------------------------------- #
def run_sweeps(seed=3, theta_T=THETA_PERIOD, n_bins=50, stem=14, arm=8):
    rng = np.random.default_rng(seed)
    W, coords, choice = t_maze(stem=stem, arm=arm, k=1.0)
    model = HippocampalModel(W, lam=0.2)
    left_states = set(range(stem, stem + arm))
    right_states = set(range(stem + arm, stem + 2 * arm))

    n_cycles = 50
    sweeps, arm_choice = [], []
    for _ in range(n_cycles):
        _, st = model.decode_sweep(choice, theta_T, n_bins, rng)
        sweeps.append(st)
        entered = [s for s in st if s in left_states or s in right_states]
        arm_choice.append(0 if not entered else
                          (-1 if entered[0] in left_states else 1))

    dist = graph_distance(W)
    start = stem - 4
    ahead = np.array([dist[start, model.decode_sweep(start, theta_T, n_bins, rng)[1]]
                      for _ in range(30)])

    Wlin, _ = linear_track(60, k_forward=1.0)
    lin = HippocampalModel(Wlin, lam=0.2)
    cyc_T, n_lin_cycles, adv = THETA_PERIOD, 12, 2.2
    track = []
    for c in range(n_lin_cycles):
        p = int(round(5 + adv * c))
        _, st = lin.decode_sweep(p, cyc_T, 40, rng)
        t0 = c * cyc_T
        track.append((t0, np.linspace(t0, t0 + cyc_T, 40), p, st))

    return dict(coords=coords, choice=choice, sweeps=sweeps, W=W,
                arm_choice=np.array(arm_choice), ahead=ahead, n_bins=n_bins,
                track=track, cyc_T=cyc_T)


# --------------------------------------------------------------------------- #
#  phase precession (linear track)
# --------------------------------------------------------------------------- #
def run_phase(N=60, f=35, k_forward=1.0, k_back=0.08, theta_T=THETA_PERIOD,
              n_phase=36, sigma=1.6, cycles=80, seed=4):
    rng = np.random.default_rng(seed)
    W, _ = linear_track(N, k_forward=k_forward, k_back=k_back)
    model = HippocampalModel(W, lam=0.2)
    L = (k_forward - k_back) * theta_T                     # per-cycle look-ahead
    positions = np.arange(int(f - L - 3), f + 3)
    positions = positions[positions >= 1]
    phases = np.linspace(0, 1, n_phase)
    # three distinct approach positions for the example within-cycle sweeps
    ex_pos = sorted({f - max(1, int(round(L * fr))) for fr in (0.85, 0.5, 0.2)})

    spike_pos, spike_phase, example_sweeps = [], [], {}
    for x_a in positions:
        for _ in range(cycles):
            _, dec = model.decode_sweep(int(x_a), theta_T, n_phase, rng)
            fired = rng.random(n_phase) < np.exp(-0.5 * ((dec - f) / sigma) ** 2)
            for ph in phases[fired]:
                spike_pos.append(x_a); spike_phase.append(ph)
        if x_a in ex_pos:
            _, dec = model.decode_sweep(int(x_a), theta_T, 200, rng)
            example_sweeps[x_a] = dec

    spike_pos = np.array(spike_pos)
    spike_phase = np.array(spike_phase) * 360.0
    fit = np.polyfit(spike_pos, spike_phase, 1)
    return dict(f=f, L=L, spike_pos=spike_pos, spike_phase=spike_phase,
                fit=fit, slope=fit[0], example_sweeps=example_sweeps)


def run():
    return dict(sw=run_sweeps(), ph=run_phase())


# --------------------------------------------------------------------------- #
#  figure: A-D forward sweeps, E-F phase precession
# --------------------------------------------------------------------------- #
def make_figure(res):
    sw, ph = res["sw"], res["ph"]
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.3)

    # (A) T-maze with forward sweeps into both arms
    axA = fig.add_subplot(gs[0, 0])
    coords, W = sw["coords"], sw["W"]
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[0]):
            if W[i, j] > 1e-9 or W[j, i] > 1e-9:
                axA.plot(coords[[i, j], 0], coords[[i, j], 1], "-",
                         color="0.88", lw=8, solid_capstyle="round", zorder=1)
    rng = np.random.default_rng(0)
    for i in list(range(0, len(sw["sweeps"]), 2))[:16]:
        st = sw["sweeps"][i]
        c = "#2166ac" if sw["arm_choice"][i] == -1 else (
            "#b2182b" if sw["arm_choice"][i] == 1 else "0.6")
        xy = coords[st] + rng.normal(0, 0.14, (len(st), 2))
        axA.plot(xy[:, 0], xy[:, 1], "-", color=c, lw=1.4, alpha=0.6, zorder=2)
    ch = coords[sw["choice"]]
    axA.plot(*ch, "^", color="k", ms=12, zorder=4)
    ytop = coords[:, 1].max()
    axA.text(coords[:, 0].min(), ytop + 1.2, "left", color="#2166ac",
             fontsize=9, fontweight="bold")
    axA.text(1.5, ytop + 1.2, "right", color="#b2182b", fontsize=9, fontweight="bold")
    axA.set_aspect("equal"); axA.axis("off"); axA.set_ylim(-1, ytop + 2.6)
    axA.set_title("A  Sweeps into both arms")

    # (B) distance ahead vs theta phase
    axB = fig.add_subplot(gs[0, 1])
    phase = np.linspace(0, 1, sw["n_bins"])
    for a in sw["ahead"][:20]:
        axB.plot(phase, a * BIN_CM, color="0.7", lw=0.8, alpha=0.6)
    axB.plot(phase, sw["ahead"].mean(0) * BIN_CM, color="#cc4c02", lw=3, label="mean")
    axB.axhline(0, color="0.5", ls="--", lw=1)
    axB.set_xlabel("theta phase (fraction of cycle)")
    axB.set_ylabel("decoded distance ahead (cm)")
    axB.set_title("B  Forward sweep per cycle")
    axB.legend(fontsize=9)

    # (C) serial arm sampling
    axC = fig.add_subplot(gs[0, 2])
    ac = sw["arm_choice"]; cyc = np.arange(len(ac))
    axC.plot(cyc, ac, "-", color="0.7", lw=1, zorder=1)
    axC.scatter(cyc[ac == -1], ac[ac == -1], color="#2166ac", s=32, label="left", zorder=2)
    axC.scatter(cyc[ac == 1], ac[ac == 1], color="#b2182b", s=32, label="right", zorder=2)
    axC.set_yticks([-1, 0, 1]); axC.set_yticklabels(["left", "stem", "right"])
    axC.set_xlabel("theta cycle")
    axC.set_title("C  Serial arm sampling")
    axC.legend(fontsize=9, loc="center right")

    # (D) linear-track theta sequences
    axD = fig.add_subplot(gs[1, 0])
    cyc_T = sw["cyc_T"]
    for (t0, tt, p, st) in sw["track"]:
        axD.plot(tt / cyc_T, np.asarray(st) * BIN_CM, color="#3690c0", lw=1.8, alpha=0.9)
        axD.plot(t0 / cyc_T, p * BIN_CM, "o", color="k", ms=4)
    ts = [t[0] / cyc_T for t in sw["track"]]; ps = [t[2] * BIN_CM for t in sw["track"]]
    axD.plot(ts, ps, "k--", lw=1.2, label="animal position")
    axD.set_xlabel("theta cycle")
    axD.set_ylabel("decoded position (cm)")
    axD.set_title("D  Theta sequences")
    axD.legend(fontsize=9, loc="upper left")

    # (E) within-cycle sweeps cross the field earlier as the animal nears it
    axE = fig.add_subplot(gs[1, 1])
    fcell = ph["f"]
    for x_a, dec in sorted(ph["example_sweeps"].items()):
        axE.plot(np.linspace(0, 360, len(dec)), np.asarray(dec) * BIN_CM, lw=2,
                 label=f"animal at {x_a * BIN_CM:.0f} cm")
    axE.axhline(fcell * BIN_CM, color="crimson", ls="--", lw=1.5, label="field center")
    axE.set_xlabel("theta phase (deg)")
    axE.set_ylabel("decoded position (cm)")
    axE.set_title("E  Theta sweeps vs field")
    axE.legend(fontsize=8, loc="upper left")

    # (F) phase precession scatter (two cycles) + linear fit
    axF = fig.add_subplot(gs[1, 2])
    sp, sph = ph["spike_pos"], ph["spike_phase"]
    jit = np.random.default_rng(0).normal(0, 0.12, sp.shape)
    for off in (0, 360):
        axF.scatter((sp + jit) * BIN_CM, sph + off, s=6, alpha=0.25, color="#1f3b73",
                    edgecolors="none")
    xx = np.array([sp.min(), sp.max()])
    axF.plot(xx * BIN_CM, ph["fit"][0] * xx + ph["fit"][1], "crimson", lw=2)
    axF.plot(xx * BIN_CM, ph["fit"][0] * xx + ph["fit"][1] + 360, "crimson", lw=2)
    axF.set_ylim(0, 720); axF.set_yticks([0, 180, 360, 540, 720])
    axF.set_xlabel("animal position (cm)")
    axF.set_ylabel("spike theta phase (deg)")
    axF.set_title("F  Phase precession")

    return fig


if __name__ == "__main__":
    res = run()
    ac = res["sw"]["arm_choice"]
    print(f"  arm sampling: left={np.mean(ac == -1):.0%}, right={np.mean(ac == 1):.0%}")
    print(f"  phase-position slope = {res['ph']['slope'] / BIN_CM:.1f} deg/cm")
    fig = make_figure(res)
    save(fig, "sim1_theta_phase.png")
