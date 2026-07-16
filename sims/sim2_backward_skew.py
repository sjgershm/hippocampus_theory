"""
Simulation 2 — Experience-dependent backward skewing of place fields.

Experiment: Mehta, Quirk & Wilson (2000), "Experience-dependent asymmetric
shape of hippocampal receptive fields", Neuron 25:707-715.
  - A rat runs repeatedly in ONE direction along a linear track.
  - Place fields start roughly symmetric and become negatively skewed
    (firing low as the rat enters the field, high as it exits), their
    center of mass shifts BACKWARD (opposite the direction of travel), and
    the fields expand.

Model account: the place field of a CA1 cell tuned to location j is the j-th
column of the resolvent R = (lam I - W)^{-1} (the continuous-time successor
representation), read out as a function of the animal's current position i.
Because the animal only moves rightward, R[i, j] is supported at i <= j (you
reach j from behind it), giving a backward tail. The recurrent generator W is
LEARNED online by the three-factor rule (Eq. 21), so the asymmetry emerges
gradually with experience -- exactly as Mehta et al. observed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import HippocampalModel, BIN_CM, THETA_PERIOD  # noqa: E402
from hippo.plotting import plt, save                     # noqa: E402


# --------------------------------------------------------------------------- #
#  field statistics (treat a non-negative field as a distribution over space)
# --------------------------------------------------------------------------- #
def field_moments(field, x):
    w = np.clip(field, 0, None)
    w = w / w.sum()
    mean = np.sum(w * x)
    var = np.sum(w * (x - mean) ** 2)
    skew = np.sum(w * (x - mean) ** 3) / (var ** 1.5 + 1e-12)
    return mean, np.sqrt(var), skew


def bump_matrix(n, sigma):
    """Row p = Gaussian EC sensory tuning centered at position p (width sigma).
    Models the finite spatial tuning of the entorhinal input."""
    x = np.arange(n)
    X = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    return X / X.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
#  run the experiment
# --------------------------------------------------------------------------- #
def run(n=40, n_laps=25, lam=0.20, eta=0.08, dt=1.0, sigma=1.6, seed=0):
    """Animal runs rightward; W is learned online lap by lap.

    EC input is a Gaussian bump, so place fields are symmetric bumps at the
    start (W=0) and develop a backward tail as the predictive generator is
    learned. The measured place field of cell j is the firing-rate map
    F[p, j] = (x_p R)[j] read out as the animal occupies position p.
    """
    x = np.arange(n, dtype=float)
    X = bump_matrix(n, sigma)                   # EC sensory tuning
    model = HippocampalModel(np.zeros((n, n)), lam=lam)   # tabula rasa

    middle = np.arange(8, n - 8)                # analyse non-boundary cells
    skew_by_lap, com_by_lap, width_by_lap = [], [], []
    snapshots = {}
    snap_laps = [1, 3, 6, 12, n_laps]

    def field_stats():
        # finite-depth (one theta cycle) predictive fields
        F = X @ model.resolvent_truncated(THETA_PERIOD)   # firing-rate maps (positions x cells)
        sk, co, wd = [], [], []
        for j in middle:
            m_, w_, s_ = field_moments(F[:, j], x)
            sk.append(s_); co.append(m_ - j); wd.append(w_)   # com relative to peak j
        return np.mean(sk), np.mean(co), np.mean(wd)

    s0 = field_stats()
    skew_by_lap.append(s0[0]); com_by_lap.append(s0[1]); width_by_lap.append(s0[2])

    for lap in range(1, n_laps + 1):
        # one rightward traversal; feedforward sensory drive clamps z to the bump
        for p in range(n - 1):
            z = X[p]
            z_dot = (X[p + 1] - X[p]) / dt
            model.learn_generator(z, z_dot, eta=eta, project=True)
        st = field_stats()
        skew_by_lap.append(st[0]); com_by_lap.append(st[1]); width_by_lap.append(st[2])
        if lap in snap_laps:
            snapshots[lap] = (X @ model.resolvent_truncated(THETA_PERIOD))[:, n // 2].copy()

    return dict(x=x, n=n, middle=middle, snap_laps=snap_laps, snapshots=snapshots,
                skew=np.array(skew_by_lap), com=np.array(com_by_lap),
                width=np.array(width_by_lap))


# --------------------------------------------------------------------------- #
#  figure (mirrors Mehta et al. 2000: example fields + skewness/COM/expansion)
# --------------------------------------------------------------------------- #
def make_figure(res):
    x, n = res["x"], res["n"]
    laps_axis = np.arange(len(res["skew"]))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (A) example place field of a central cell across laps
    j = n // 2
    xc = (x - j) * BIN_CM                        # position relative to peak (cm)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(res["snap_laps"])))
    for c, lap in zip(cmap, res["snap_laps"]):
        f = res["snapshots"][lap]
        ax[0, 0].plot(xc, f / f.max(), color=c, label=f"lap {lap}")
    ax[0, 0].axvline(0, color="0.6", ls="--", lw=1)
    ax[0, 0].annotate("", xy=(44, 1.09), xytext=(8, 1.09),
                      arrowprops=dict(arrowstyle="->", color="k"),
                      annotation_clip=False)
    ax[0, 0].text(26, 1.13, "direction of travel", fontsize=9, ha="center",
                  va="bottom", clip_on=False)
    ax[0, 0].set_ylim(top=1.22)
    ax[0, 0].set_xlabel("position relative to field peak (cm)")
    ax[0, 0].set_ylabel("normalized firing rate")
    ax[0, 0].set_title("A  Backward-skewing field")
    ax[0, 0].legend(fontsize=9, loc="upper left")
    ax[0, 0].text(-56, 0.5, "backward\n(behind animal)", fontsize=8, color="0.4",
                  ha="center")

    # (B) skewness vs lap  (becomes negative)
    ax[0, 1].plot(laps_axis, res["skew"], "-o", color="#c0392b", ms=4)
    ax[0, 1].axhline(0, color="0.6", ls="--", lw=1)
    ax[0, 1].set_xlabel("lap number")
    ax[0, 1].set_ylabel("place-field skewness")
    ax[0, 1].set_title("B  Skewness")

    # (C) center-of-mass shift vs lap (backward = negative)
    ax[1, 0].plot(laps_axis, res["com"] * BIN_CM, "-o", color="#2c3e50", ms=4)
    ax[1, 0].axhline(0, color="0.6", ls="--", lw=1)
    ax[1, 0].set_xlabel("lap number")
    ax[1, 0].set_ylabel("center-of-mass shift (cm)")
    ax[1, 0].set_title("C  Center-of-mass shift")

    # (D) field width vs lap (expansion)
    ax[1, 1].plot(laps_axis, res["width"] * BIN_CM, "-o", color="#27ae60", ms=4)
    ax[1, 1].set_xlabel("lap number")
    ax[1, 1].set_ylabel("place-field width (s.d., cm)")
    ax[1, 1].set_title("D  Field width")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    res = run()
    print(f"  skewness  lap0={res['skew'][0]:+.3f} -> lap{len(res['skew'])-1}={res['skew'][-1]:+.3f}")
    print(f"  COM shift lap0={res['com'][0]*BIN_CM:+.2f} -> lap{len(res['com'])-1}={res['com'][-1]*BIN_CM:+.2f} cm")
    print(f"  width     lap0={res['width'][0]*BIN_CM:.2f}  -> lap{len(res['width'])-1}={res['width'][-1]*BIN_CM:.2f} cm")
    fig = make_figure(res)
    save(fig, "sim2_backward_skew.png")
