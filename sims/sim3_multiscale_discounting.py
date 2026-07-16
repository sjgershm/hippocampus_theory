"""
Simulation 3 — One leak-rate distribution, two readouts: multi-scale place
fields and hyperbolic discounting.

The leak rates are taken from CA1 data: lambda = 1/tau for the 32 cells best fit
by exponential decay (doi:10.1002/hipo.70105, SI). The time constants tau are
exponentially distributed, so the same population is read out two ways:

  - SPATIAL (panels A-C): each cell's steady-state field width falls with the
    leak rate, width = c/sqrt(lambda) = c*sqrt(tau) (screened Green's function,
    DERIVATIONS.md). Pushing the leak-rate distribution through this relation
    (a single spatial-scale parameter c) predicts the distribution of tuning
    widths. Because tau is exponential this pushforward is a Rayleigh/Weibull-2
    law, which is compared in panel C to the aggregated empirical widths from
    Cohen & Drugowitsch (2025) (Pfeiffer & Foster 2013 dataset, field_widths/).
  - TEMPORAL (panels D-E): the discount function is the Laplace transform of the
    SAME leak-rate distribution -- approximately hyperbolic 1/(1 + mu t), far
    more sustained than any single timescale, producing preference reversals
    (Momennejad & Howard 2018; Masset et al. 2025).

Spatial scale: BIN_CM cm per bin (hippo.BIN_CM), used consistently everywhere.
The steady-state (asymptotic) width is used here -- the intrinsic tuning scale --
while the behavioral SR simulations use the 125 ms truncated resolvent.

See DERIVATIONS.md for the math (and the Fréchet vs Gamma comparison, §3.7).
"""

import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hippo import HippocampalModel, BIN_CM                 # noqa: E402
from hippo.environments import open_arena                  # noqa: E402
from hippo.plotting import plt, save, panel_label          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Time constants (tau, seconds) of CA1 cells best fit by exponential decay, from
# the SI of doi:10.1002/hipo.70105. Leak rate lambda = 1/tau (per second).
CA1_TAU = np.array([1.02, 1.81, 2.04, 2.11, 2.47, 6.61, 13.18, 13.90, 14.11,
                    14.74, 25.45, 34.11, 49.22, 1.59, 3.19, 3.71, 5.74, 6.45,
                    7.01, 7.21, 7.74, 8.71, 9.40, 20.88, 26.72, 27.55, 29.44,
                    31.29, 7.76, 12.51, 12.52, 26.84])


def field_radius(field, coords, com=None):
    w = np.clip(field, 0, None)
    w = w / w.sum()
    if com is None:
        com = w @ coords
    return np.sqrt(np.sum(w * ((coords - com) ** 2).sum(axis=1))), com


def load_empirical_widths():
    """Aggregate the per-session tuning-width samples (cm) into one population."""
    with open(os.path.join(_ROOT, "field_widths", "field_widths.pkl"), "rb") as f:
        d = pickle.load(f)
    return np.concatenate([np.asarray(v, float).ravel() for v in d.values()])


# --------------------------------------------------------------------------- #
#  spatial readout: place-field widths (asymptotic / steady-state scale)
# --------------------------------------------------------------------------- #
def run_widths(nx=52, ny=44, seed=1):
    # Example fields and the width-vs-leak-rate relation, measured from the
    # (asymptotic) resolvent of a 2D open arena. Widths reported in cm.
    W, coords, shape = open_arena(nx, ny)
    model = HippocampalModel(W, lam=0.2)
    lam_grid = np.geomspace(0.02, 4.0, 45)
    R = {lam: model.resolvent(lam=lam) for lam in lam_grid}
    nxs, nys = shape
    center = (nxs // 2) * nys + (nys // 2)
    w_cm = BIN_CM * np.array([field_radius(R[lam][:, center], coords)[0]
                              for lam in lam_grid])

    # example fields spanning the empirical range (small / typical / large)
    examples = []
    for tw in (6.0, 14.0, 28.0):
        j = int(np.argmin(np.abs(w_cm - tw)))
        examples.append((w_cm[j], R[lam_grid[j]][:, center].reshape(nxs, nys)))

    return dict(shape=shape, lam_grid=lam_grid, w_cm=w_cm, examples=examples)


# --------------------------------------------------------------------------- #
#  spatial readout: predicted width distribution (pushforward of leak rates)
# --------------------------------------------------------------------------- #
def predict_widths(w_emp, tau=CA1_TAU, n=200000, seed=0):
    """Push the leak-rate distribution through width = c/sqrt(lambda).

    lambda = 1/tau with tau ~ Exponential (the CA1 data), so width = c*sqrt(tau)
    and the pushforward is Rayleigh/Weibull-2. The single spatial-scale c is
    calibrated to the empirical median (one free parameter); the SHAPE is
    predicted, and with it the mean / upper percentiles.
    """
    rng = np.random.default_rng(seed)
    tsamp = rng.exponential(tau.mean(), n)
    c = np.median(w_emp) / np.median(np.sqrt(tsamp))        # cm * s^(1/2)
    sigma = c * np.sqrt(tau.mean() / 2.0)                    # Rayleigh scale (cm)
    return dict(c=c, sigma=sigma,
                w_pred=c * np.sqrt(tsamp),
                mu_lam=float((1.0 / tau).mean()))


# --------------------------------------------------------------------------- #
#  temporal readout: discounting (Laplace transform of the same leak rates)
# --------------------------------------------------------------------------- #
def run_discount(tau=CA1_TAU, n_lam=200000, seed=0):
    # bank drawn from the SAME leak-rate distribution (lambda = 1/tau, tau~Exp);
    # times in seconds since lambda is per second.
    rng = np.random.default_rng(seed)
    lambdas = 1.0 / rng.exponential(tau.mean(), n_lam)
    mu = float((1.0 / tau).mean())
    t = np.linspace(0, 8.0 / mu, 400)

    def D(tt):
        return np.exp(-np.outer(lambdas, tt)).mean(0)

    A_s, d_s, A_l, d_l = 1.0, 0.5 / mu, 1.6, 2.0 / mu
    adv = np.linspace(0, 6.0 / mu, 300)
    return dict(t=t, D_bank=D(t), D_hyper=1.0 / (1.0 + mu * t), D_exp=np.exp(-mu * t),
                mu=mu, adv=adv,
                V_ss_bank=A_s * D(d_s + adv), V_ll_bank=A_l * D(d_l + adv),
                V_ss_exp=A_s * np.exp(-mu * (d_s + adv)),
                V_ll_exp=A_l * np.exp(-mu * (d_l + adv)))


def run():
    w_emp = load_empirical_widths()
    return dict(wd=run_widths(), pr=predict_widths(w_emp), dc=run_discount(),
                w_emp=w_emp, lam_emp=1.0 / CA1_TAU)


# --------------------------------------------------------------------------- #
#  figure: A-C spatial widths, D-E temporal discounting
# --------------------------------------------------------------------------- #
def make_figure(res):
    wd, pr, dc = res["wd"], res["pr"], res["dc"]
    nx, ny = wd["shape"]
    w_emp, lam_emp = res["w_emp"], res["lam_emp"]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.82, 1, 1], hspace=0.5, wspace=1.1)

    # (A) example fields at three widths (cm)
    for k, (w, f) in enumerate(wd["examples"]):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        extent = [0, nx * BIN_CM, 0, ny * BIN_CM]
        ax.imshow(f.T / f.max(), origin="lower", cmap="jet", aspect="equal", extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"width ≈ {w:.0f} cm", fontsize=10)
        if k == 0:
            panel_label(ax, "A")

    # (B) field width vs leak rate: model measurements follow width = c/sqrt(lam)
    axB = fig.add_subplot(gs[1, 0:3])
    c = pr["c"]
    lam_arena = wd["lam_grid"]
    # place the arena measurements on the physical leak-rate axis (one global
    # scale = the arena's diffusion coefficient, set by the calibration c)
    c_arena = np.median(wd["w_cm"] * np.sqrt(lam_arena))
    lam_phys = lam_arena * (c / c_arena) ** 2
    ll = np.geomspace(lam_phys.min(), lam_phys.max(), 200)
    axB.plot(ll, c / np.sqrt(ll), color="#1f6f8b", lw=2.3,
             label=r"$w = c/\sqrt{\lambda}$")
    axB.plot(lam_phys, wd["w_cm"], "o", color="#1f3b73", ms=3.5, label="model fields")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel(r"leak rate $\lambda$ (s$^{-1}$)")
    axB.set_ylabel("place-field width (cm)")
    axB.set_title("B  Field width vs leak rate")
    axB.legend(fontsize=8.5, loc="upper right")

    # (C) empirical width distribution vs prediction from the leak-rate density
    axC = fig.add_subplot(gs[1, 3:6])
    bins = np.linspace(0, np.percentile(w_emp, 99), 30)
    axC.hist(w_emp, bins=bins, density=True, color="#cfe8c5", alpha=0.95,
             label="CA1 data")
    xx = np.linspace(0, bins[-1], 300)
    axC.plot(xx, stats.rayleigh.pdf(xx, scale=pr["sigma"]), color="#1f6f8b", lw=2.3,
             label=r"prediction ($w\!=\!c/\sqrt{\lambda}$)")
    axC.set_xlabel("place-field width (cm)")
    axC.set_ylabel("probability density")
    axC.set_title("C  Width distribution")
    axC.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.43),
               framealpha=0.9)
    # inset: the leak-rate distribution (lambda = 1/tau) with an INVERSE-exponential
    # fit (tau = 1/lambda ~ exponential, mean mu_tau). Its pushforward through
    # w = c/sqrt(lambda) is exactly the Rayleigh curve in the main panel, so the
    # inset and the width prediction are strictly one law. (A plain exponential fit
    # to lambda would push forward to a Frechet-2 that fits the widths poorly,
    # KS 0.28 vs 0.02.)
    mu_tau = CA1_TAU.mean()
    axin = axC.inset_axes([0.60, 0.57, 0.37, 0.38])
    axin.hist(lam_emp, bins=np.linspace(0, 1, 9), density=True, color="#9ecae1",
              edgecolor="white")
    xl = np.linspace(0.01, 1, 300)
    axin.plot(xl, np.exp(-1.0 / (xl * mu_tau)) / (mu_tau * xl ** 2), "k--", lw=1.6)
    axin.set_xlabel(r"leak rate $\lambda$", fontsize=8)
    axin.set_ylabel("density", fontsize=8)
    axin.tick_params(labelsize=7); axin.set_yticks([])

    # (D) discounting: bank = Laplace transform of p(lambda) ~ hyperbolic
    axD = fig.add_subplot(gs[2, 0:3])
    axD.plot(dc["t"], dc["D_exp"], color="#d95f0e", lw=2.3, label=r"single $\lambda$ (exp.)")
    axD.plot(dc["t"], dc["D_bank"], color="#1f6f8b", lw=3, label="bank (Laplace)")
    axD.plot(dc["t"], dc["D_hyper"], "k--", lw=1.6, label="hyperbolic")
    axD.set_xlabel("delay $t$ (s)"); axD.set_ylabel("discount factor")
    axD.set_title("D  Hyperbolic discounting")
    axD.legend(fontsize=8.5, loc="upper right")

    # (E) preference reversal
    axE = fig.add_subplot(gs[2, 3:6])
    adv = dc["adv"]
    axE.plot(adv, dc["V_ss_exp"], color="#fdae6b", lw=2, ls=":", label="smaller-sooner (exp)")
    axE.plot(adv, dc["V_ll_exp"], color="#a63603", lw=2, ls=":", label="larger-later (exp)")
    axE.plot(adv, dc["V_ss_bank"], color="#9ecae1", lw=2.5, label="smaller-sooner (bank)")
    axE.plot(adv, dc["V_ll_bank"], color="#08519c", lw=2.5, label="larger-later (bank)")
    cross = np.where(np.diff(np.sign(dc["V_ll_bank"] - dc["V_ss_bank"])))[0]
    if len(cross):
        xc = adv[cross[0]]
        axE.axvline(xc, color="0.5", ls="--", lw=1)
        axE.annotate("preference\nreversal", xy=(xc, dc["V_ss_bank"][cross[0]]),
                     xytext=(xc + 0.18 * adv.max(), dc["V_ss_bank"][cross[0]] + 0.12),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
    axE.set_xlabel("time before the sooner reward (s)")
    axE.set_ylabel("discounted value")
    axE.set_title("E  Preference reversal")
    axE.legend(fontsize=7.5, loc="upper right")

    return fig


if __name__ == "__main__":
    res = run()
    w_emp, pr = res["w_emp"], res["pr"]
    wp = pr["w_pred"]
    print(f"  empirical widths  (n={len(w_emp)}): median {np.median(w_emp):.1f}, "
          f"mean {w_emp.mean():.1f}, p95 {np.percentile(w_emp,95):.1f} cm")
    print(f"  predicted widths  (c={pr['c']:.2f} cm·s^0.5): median {np.median(wp):.1f}, "
          f"mean {wp.mean():.1f}, p95 {np.percentile(wp,95):.1f} cm")
    err = np.abs(res["dc"]["D_bank"] - res["dc"]["D_hyper"]).max()
    print(f"  discounting from CA1 leak rates 1/tau (mu={res['dc']['mu']:.3f}): "
          f"bank vs hyperbolic max diff = {err:.4f}")
    fig = make_figure(res)
    save(fig, "sim3_multiscale_discounting.png")
