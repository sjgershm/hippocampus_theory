"""
Core network model from:

    Gershman, S. J. "Prediction, imagination, memory: variations on a
    hippocampal theme."

A linear recurrent network with three populations:

    x            input            (entorhinal cortex, EC)
    z(t)         internal state   (CA3, recurrent)
    y(t)         output           (CA1, leaky integrator)

Dynamics (Eqs. 1-2), with z, y treated as ROW vectors:

    tau * zdot(t) =  z(t) W                         (Eq. 1)
    tau * ydot(t) = -lam * y(t) + z(t)              (Eq. 2)

with z(0) = x, y(0) = 0, so the internal state is the propagator applied
to the input (Eq. 3):

    z(t) = x exp((t/tau) W).

W is the *generator* of a continuous-time Markov chain (master equation),
satisfying (footnote 2):

    W_ij >= 0   (i != j),    W_ii <= 0,    sum_j W_ij = 0.

Key derived objects:

    propagator         exp((t/tau) W)            maps a belief to a future belief
    resolvent          (lam I - W)^{-1}          continuous-time successor
                                                 representation / predictive map (Eq. 8)
    value              x (lam I - W)^{-1} r       discounted return (Eq. 13)
    Laplace bank       {(lam_m I - W)^{-1}}_m     multi-scale predictive map (Eq. 16)
    stochastic rollout sampled CTMC trajectory   "imagination" on a theta cycle (Eq. 5)

The leak rate lam plays the role of the RL discount factor (lam ~ -log gamma);
larger lam == steeper discounting == shorter-range / smaller place fields.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------------- #
#  Generator utilities
# ----------------------------------------------------------------------------- #
def project_to_generator(W: np.ndarray) -> np.ndarray:
    """Project an arbitrary matrix onto a valid CTMC generator (footnote 2).

    Off-diagonal entries are clipped to be non-negative (they are transition
    *rates*); each diagonal entry is set to minus the sum of its row's
    off-diagonal entries so that every row sums to zero.

    This is applied to the *ungated* recurrent weights `W` (the `b = 1` case
    used by the environments and the ungated online learner). Under DG gating
    the generator that actually governs the dynamics is the product
    `J_ij = W_ij b_j` (Eq. 18), which is a *different* matrix; its generator
    structure is enforced separately in
    `ContextHippocampalModel.effective_generator`, which is exactly this same
    projection applied to the gated off-diagonal product. Projecting `W` alone
    does not make `J` a generator (its rows would not sum to zero for
    non-uniform `b`) — the diagonal must be re-derived after gating, which is
    why the gated case is handled there rather than here.
    """
    J = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(J, 0.0)
    J[J < 0] = 0.0
    np.fill_diagonal(J, -J.sum(axis=1))
    return J


def is_generator(W: np.ndarray, tol: float = 1e-8) -> bool:
    off = W - np.diag(np.diag(W))
    return (off.min() >= -tol) and (np.abs(W.sum(axis=1)).max() <= tol)


# ----------------------------------------------------------------------------- #
#  The model
# ----------------------------------------------------------------------------- #
class HippocampalModel:
    """The single-context network (EC -> CA3 -> CA1)."""

    def __init__(self, W: np.ndarray, lam: float = 0.2, tau: float = 1.0):
        self.W = np.asarray(W, dtype=float)
        self.D = self.W.shape[0]
        self.lam = float(lam)
        self.tau = float(tau)

    # ---- imagination: propagator and internal-state dynamics (Eqs. 1, 3) ---- #
    def propagator(self, t: float, gen: np.ndarray | None = None) -> np.ndarray:
        """exp((t/tau) W): linear map from current to future internal state."""
        W = self.W if gen is None else gen
        return expm((t / self.tau) * W)

    def internal_state(self, x: np.ndarray, t: float,
                       gen: np.ndarray | None = None) -> np.ndarray:
        """z(t) = x exp((t/tau) W)  (Eq. 3): the imagined future belief."""
        return np.asarray(x, float) @ self.propagator(t, gen)

    # ---- prediction: resolvent / successor representation (Eqs. 8, 13) ------ #
    def resolvent(self, lam: float | None = None,
                  gen: np.ndarray | None = None) -> np.ndarray:
        """(lam I - W)^{-1}: continuous-time successor representation.

        Entry [i, j] is the discounted future occupancy of state j given the
        process starts in state i:  integral_0^inf e^{-lam t} P_ij(t) dt.
        """
        lam = self.lam if lam is None else lam
        W = self.W if gen is None else gen
        return np.linalg.inv(lam * np.eye(W.shape[0]) - W)

    def resolvent_truncated(self, T: float, lam: float | None = None,
                            gen: np.ndarray | None = None) -> np.ndarray:
        """Finite-depth (truncated) resolvent, matrix form of Eq. 14:

            R_T = (lam I - W)^{-1} (I - exp(-(lam I - W) T))
                = integral_0^T e^{-lam t} e^{W t} dt.

        The continuous-time successor representation accumulated over a single
        horizon T (one theta cycle) instead of the infinite horizon. Column j is
        the finite-depth predictive place field of cell j: it keeps local
        structure and suppresses the slow/global modes that only a full sweep
        would reach. For T -> inf this recovers the ordinary resolvent.
        """
        lam = self.lam if lam is None else lam
        W = self.W if gen is None else gen
        A = lam * np.eye(W.shape[0]) - W
        return np.linalg.inv(A) @ (np.eye(W.shape[0]) - expm(-A * T))

    def output(self, x: np.ndarray, lam: float | None = None,
               gen: np.ndarray | None = None) -> np.ndarray:
        """y(inf) = x (lam I - W)^{-1}  (Eq. 8)."""
        return np.asarray(x, float) @ self.resolvent(lam, gen)

    def output_finite_depth(self, x: np.ndarray, T: float,
                            lam: float | None = None,
                            gen: np.ndarray | None = None) -> np.ndarray:
        """Limited-depth resolvent (Eq. 14):

            y(T) = x (lam I - W)^{-1} (I - exp(-(lam I - W) T)).

        For T -> inf this recovers y(inf); for finite T it acts as a spectral
        filter that preserves local (high-frequency) structure.
        """
        lam = self.lam if lam is None else lam
        W = self.W if gen is None else gen
        A = lam * np.eye(W.shape[0]) - W
        R = np.linalg.inv(A)
        return np.asarray(x, float) @ (R @ (np.eye(W.shape[0]) - expm(-A * T)))

    def value(self, x: np.ndarray, r: np.ndarray,
              lam: float | None = None) -> float:
        """V(x) = x (lam I - W)^{-1} r  (Eq. 13)."""
        return float(np.asarray(x, float) @ self.resolvent(lam) @ np.asarray(r, float))

    # ---- multi-scale prediction: the Laplace transform (Eqs. 15-16) -------- #
    def laplace_bank(self, x: np.ndarray, lambdas) -> np.ndarray:
        """Bank of leaky integrators across leak rates.

        Returns an (M, D) array whose m-th row is y_{lam_m}(inf) = x (lam_m I - W)^{-1}.
        Densely sampling lambdas approximates the Laplace transform of z(t)
        and yields a *multi-scale* predictive map.
        """
        return np.array([self.output(x, lam) for lam in lambdas])

    # ---- spectral analysis (Eqs. 6-7) -------------------------------------- #
    def spectrum(self):
        """Eigenvalues/vectors of W. For a generator the leading eigenvalue is
        0 (stationary distribution); the rest have negative real part."""
        vals, vecs = np.linalg.eig(self.W.T)  # left eigvecs of W = right of W.T
        order = np.argsort(-vals.real)
        return vals[order], vecs[:, order]

    # ---- imagination: stochastic rollouts (Eq. 5, Monte-Carlo sweeps) ------ #
    def sample_trajectory(self, s0: int, T: float, rng: np.random.Generator,
                          gen: np.ndarray | None = None):
        """Gillespie sample of a CTMC path with generator W, from state s0 over
        [0, T]. Returns (jump_times, states): a single coherent realization
        z-tilde_k (the kind decoded on one theta cycle)."""
        W = self.W if gen is None else gen
        D = W.shape[0]
        t, s = 0.0, int(s0)
        times, states = [0.0], [s]
        while True:
            rate = -W[s, s]
            if rate <= 1e-12:                      # absorbing state -> dwell
                break
            t += rng.exponential(1.0 / rate)
            if t > T:
                break
            p = W[s].copy()
            p[s] = 0.0
            p[p < 0] = 0.0
            tot = p.sum()
            if tot <= 0:
                break
            s = int(rng.choice(D, p=p / tot))
            times.append(min(t, T))
            states.append(s)
        times.append(T)
        states.append(s)
        return np.asarray(times), np.asarray(states)

    def decode_sweep(self, s0: int, T: float, n_bins: int,
                     rng: np.random.Generator, gen: np.ndarray | None = None):
        """Decoded state at n_bins evenly spaced times in [0, T] for one rollout.
        This is the within-theta-cycle forward sweep."""
        times, states = self.sample_trajectory(s0, T, rng, gen)
        grid = np.linspace(0.0, T, n_bins)
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1,
                      0, len(states) - 1)
        return grid, states[idx]

    def mc_belief(self, s0: int, T: float, K: int, n_bins: int,
                  rng: np.random.Generator, gen: np.ndarray | None = None):
        """Monte-Carlo estimate of z(t) by averaging K rollouts (Eq. 5):
        z(t) ~= (1/K) sum_k z-tilde_k(t). Returns (grid, belief[n_bins, D])."""
        D = (self.W if gen is None else gen).shape[0]
        grid = np.linspace(0.0, T, n_bins)
        belief = np.zeros((n_bins, D))
        for _ in range(K):
            _, st = self.decode_sweep(s0, T, n_bins, rng, gen)
            belief[np.arange(n_bins), st] += 1.0
        return grid, belief / K

    # ---- learning: three-factor generator rule (Eqs. 20-21) ---------------- #
    def learn_generator(self, z: np.ndarray, z_dot: np.ndarray,
                        b: np.ndarray | None = None, eta: float = 0.01,
                        project: bool = True) -> np.ndarray:
        """One online gradient step on the recurrent weights.

            eps_j   = zdot_j - sum_i z_i W_ij b_j                  (Eq. 20)
            dW_ij  ~= eps_j b_j z_i                                (Eq. 21)

        A three-factor Hebbian rule: presynaptic z_i, postsynaptic dendritic
        error eps_j, and DG gating term b_j. Returns the prediction error.
        """
        if b is None:
            b = np.ones(self.D)
        pred = (z @ self.W) * b
        eps = z_dot - pred
        self.W += eta * np.outer(z, eps * b)
        if project:
            self.W = project_to_generator(self.W)
        return eps


# ----------------------------------------------------------------------------- #
#  Context-gated model (DG -> CA3)
# ----------------------------------------------------------------------------- #
class ContextHippocampalModel(HippocampalModel):
    """CA3 recurrent dynamics gated by a dentate-gyrus context vector c.

        tau zdot_j = sum_i z_i W_ij b_j(t)                        (Eq. 17)
        b_j(t)     = sum_m c_m(t) G_mj                            (Eq. 18)

    The active generator is the Jacobian J_ij = W_ij b_j, re-balanced so each
    row sums to zero (lateral inhibition). Different contexts c route CA3 to
    different effective generators -> different CA1 maps -> remapping.
    """

    def __init__(self, W: np.ndarray, G: np.ndarray, lam: float = 0.2,
                 tau: float = 1.0):
        super().__init__(W, lam=lam, tau=tau)
        self.G = np.asarray(G, dtype=float)          # (M_context, D)
        self.M = self.G.shape[0]

    def gating(self, c: np.ndarray) -> np.ndarray:
        """b = c G  (Eq. 18): per-column multiplicative gate, b_j >= 0."""
        return np.asarray(c, float) @ self.G

    def effective_generator(self, c: np.ndarray) -> np.ndarray:
        """Gated generator `J_ij = W_ij b_j` (Eq. 18), projected to a valid CTMC.

        Gate the off-diagonal transition *rates* by the destination gate `b_j`
        (`J_ij = W_ij b_j`, i != j), then set the diagonal
        `J_ii = -sum_{j != i} W_ij b_j` so every row sums to zero. This is
        precisely `project_to_generator` applied to the gated off-diagonal
        product — i.e. the generator constraint IS enforced on `W_ij b_j`, not
        just on `W`. The diagonal must be *re-derived* (not scaled by `b`):
        the literal all-entries product `W_ij b_j` is a generator only when `b`
        is uniform, otherwise its rows do not sum to zero and probability is not
        conserved.
        """
        b = self.gating(c)
        J = self.W.copy()
        np.fill_diagonal(J, 0.0)                      # keep only off-diagonal rates
        J[J < 0] = 0.0
        J = J * b[None, :]                            # J_ij = W_ij b_j  (gate destination j)
        np.fill_diagonal(J, -J.sum(axis=1))           # re-derive diagonal -> rows sum to zero
        return J

    def resolvent_ctx(self, c: np.ndarray, lam: float | None = None,
                      T: float | None = None) -> np.ndarray:
        """Context-gated successor representation.

        With `T` given, uses the finite-depth (one-theta-cycle) resolvent
        (Eq. 14); otherwise the infinite-horizon resolvent.
        """
        J = self.effective_generator(c)
        if T is None:
            return self.resolvent(lam, gen=J)
        return self.resolvent_truncated(T, lam, gen=J)

    def output_ctx(self, x: np.ndarray, c: np.ndarray,
                   lam: float | None = None, T: float | None = None) -> np.ndarray:
        """CA1 output for input x under context c (finite depth if T given)."""
        J = self.effective_generator(c)
        if T is None:
            return self.output(x, lam, gen=J)
        return np.asarray(x, float) @ self.resolvent_truncated(T, lam, gen=J)

    def learn_gating(self, z: np.ndarray, eps: np.ndarray, c: np.ndarray,
                     eta: float = 0.01) -> None:
        """Online rule for the DG->CA3 gating weights (Eq. 22):

            dG_mj ~= eps_j c_m sum_i z_i W_ij.
        """
        zW = z @ self.W
        self.G += eta * np.outer(c, eps * zW)
        self.G[self.G < 0] = 0.0
