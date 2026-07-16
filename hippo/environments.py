"""
Environments / behavioral policies, expressed as CTMC generators W.

Each builder returns a generator W (rows sum to 0, off-diagonals are transition
*rates*) together with spatial coordinates for plotting. The generator encodes
the animal's behavioral policy (direction of travel, dwell times, branching at
choice points), which is what shapes the predictive map.
"""

from __future__ import annotations

import numpy as np

from .model import project_to_generator


# --------------------------------------------------------------------------- #
#  1D linear track (Mehta et al., 2000)
# --------------------------------------------------------------------------- #
def linear_track(n: int, k_forward: float = 1.0, k_back: float = 0.0):
    """Directed run along a 1D track of n states.

    Rightward rate k_forward, leftward rate k_back (0 -> deterministic forward).
    Returns (W, coords) with coords[i] = i (the position in bins).
    """
    W = np.zeros((n, n))
    for i in range(n):
        if i + 1 < n:
            W[i, i + 1] += k_forward
        if i - 1 >= 0:
            W[i, i - 1] += k_back
    W = project_to_generator(W)
    return W, np.arange(n, dtype=float)


# --------------------------------------------------------------------------- #
#  Annular watermaze (Hollup et al., 2001)
# --------------------------------------------------------------------------- #
def annular_track(n: int, goal: int, drift: float = 1.0, diffusion: float = 0.15,
                  dwell_factor: float = 1.0, dwell_w: float = 6.0):
    """Ring of n states with a goal-directed swimming policy.

    From each state the animal drifts along the *shorter* arc toward `goal`
    (rate `drift`), with symmetric diffusion (`diffusion`). The animal slows
    smoothly as it nears the platform: rates are scaled by a factor that drops
    to `dwell_factor` at the goal and relaxes back to 1 over a Gaussian width
    `dwell_w` (in states). `dwell_factor = 1` means no dwelling. The smooth
    profile (vs. a sharp dwell zone) puts the representation peak ON the goal
    rather than at a dwell-zone boundary.

    Returns (W, angles) with angles in radians (goal at angle 0).
    """
    W = np.zeros((n, n))
    half = n / 2.0
    for i in range(n):
        cw = (i + 1) % n                          # clockwise neighbour
        ccw = (i - 1) % n                         # counter-clockwise neighbour
        signed = ((goal - i + half) % n) - half   # signed arc distance to goal
        toward = cw if signed > 0 else ccw        # neighbour toward the goal
        away = ccw if signed > 0 else cw
        dist = min(abs(i - goal), n - abs(i - goal))
        slow = dwell_factor + (1 - dwell_factor) * (1 - np.exp(-dist ** 2 / (2 * dwell_w ** 2)))
        W[i, toward] += slow * (drift + diffusion)
        W[i, away] += slow * diffusion
    W = project_to_generator(W)
    angles = (np.arange(n) - goal) * (2 * np.pi / n)
    angles = (angles + np.pi) % (2 * np.pi) - np.pi   # wrap to (-pi, pi], goal=0
    return W, angles


def annular_track_undirected(n: int, diffusion: float = 1.0):
    """Pre-training control: unbiased random walk around the ring (no goal)."""
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] += diffusion
        W[i, (i - 1) % n] += diffusion
    return project_to_generator(W), (np.arange(n)) * (2 * np.pi / n)


# --------------------------------------------------------------------------- #
#  T-maze with a choice point (Johnson & Redish, 2007)
# --------------------------------------------------------------------------- #
def t_maze(stem: int = 12, arm: int = 9, k: float = 1.0):
    """A T-maze: vertical stem -> choice point -> left and right arms.

    State ordering: [stem_0 .. stem_{stem-1}=choice point,
                     left_1 .. left_arm, right_1 .. right_arm].
    At the choice point the animal can enter either arm (branching).

    Returns (W, coords, choice_idx) where coords is (N, 2) for plotting.
    """
    n = stem + 2 * arm
    W = np.zeros((n, n))
    coords = np.zeros((n, 2))

    # stem: positions (0, 0) .. (0, stem-1); run upward
    for i in range(stem):
        coords[i] = (0.0, i)
        if i + 1 < stem:
            W[i, i + 1] += k
    choice = stem - 1                              # choice point (top of stem)

    left0 = stem
    right0 = stem + arm
    # branch from choice point into both arms
    W[choice, left0] += k
    W[choice, right0] += k
    # left arm: extends to negative x at height of choice point
    for j in range(arm):
        idx = left0 + j
        coords[idx] = (-(j + 1), choice)
        if j + 1 < arm:
            W[idx, idx + 1] += k
    # right arm: extends to positive x
    for j in range(arm):
        idx = right0 + j
        coords[idx] = (j + 1, choice)
        if j + 1 < arm:
            W[idx, idx + 1] += k

    W = project_to_generator(W)
    return W, coords, choice


# --------------------------------------------------------------------------- #
#  2D open arena / "megaspace" (Harland et al., 2021)
# --------------------------------------------------------------------------- #
def open_arena(nx: int, ny: int, k: float = 1.0):
    """2D grid with an unbiased (diffusive) random-walk policy.

    Returns (W, coords, shape) with coords[i] = (x, y) and shape = (nx, ny).
    """
    n = nx * ny
    W = np.zeros((n, n))
    coords = np.zeros((n, 2))

    def idx(ix, iy):
        return ix * ny + iy

    for ix in range(nx):
        for iy in range(ny):
            i = idx(ix, iy)
            coords[i] = (ix, iy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    W[i, idx(jx, jy)] += k
    W = project_to_generator(W)
    return W, coords, (nx, ny)


# --------------------------------------------------------------------------- #
#  Figure-8 continuous-alternation maze (Wood et al. 2000; Frank et al. 2000)
# --------------------------------------------------------------------------- #
def figure_eight(stem: int = 10, loop: int = 14, k: float = 1.0):
    """A figure-8 / continuous T-alternation maze.

    The animal runs up a shared central STEM to a choice point, takes the left
    or right return LOOP, and arrives back at the stem base -- a continuous
    cycle. The stem is common to both trajectory types, so trajectory-dependent
    ("splitter") firing there cannot come from current location alone.

    Returns a dict with the baseline generator `W` (both loops open), `coords`,
    and index arrays: `stem` (bottom->choice), `choice`, `left`, `right`
    (loop states, each running choice -> back to stem base).
    """
    n = stem + 2 * loop
    W = np.zeros((n, n))
    coords = np.zeros((n, 2))

    stem_idx = np.arange(stem)
    for i in range(stem):
        coords[i] = (0.0, i)
        if i + 1 < stem:
            W[i, i + 1] += k
    choice = stem - 1

    left = np.arange(stem, stem + loop)
    right = np.arange(stem + loop, stem + 2 * loop)
    # semicircular loops from the choice point back down to the stem base (0,0)
    for s, idx_arr, side in ((-1, left, "L"), (1, right, "R")):
        W[choice, idx_arr[0]] += k
        for j, idx in enumerate(idx_arr):
            frac = (j + 1) / (loop + 1)
            ang = np.pi * frac                       # sweep from choice to base
            coords[idx] = (s * 3.4 * np.sin(ang), (stem - 1) * (1 + np.cos(ang)) / 2)
            nxt = idx_arr[j + 1] if j + 1 < loop else 0   # last -> stem base
            W[idx, nxt] += k

    W = project_to_generator(W)
    return dict(W=W, coords=coords, stem=stem_idx, choice=choice,
                left=left, right=right, n=n)
