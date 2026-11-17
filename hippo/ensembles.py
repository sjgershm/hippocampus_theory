"""
Shared ingredients for the context-gating simulations (Figs. 4 and 5).

These are used identically by Simulation 4 (DG-gated context states), Simulation 5
(partial vs complete remapping) and its morph-frequency component, so they live
here rather than being duplicated in each.
"""

from __future__ import annotations

import numpy as np


def context_gates(n: int, rng: np.random.Generator, frac: float = 0.6,
                  lo: float = 0.05) -> np.ndarray:
    """Two sparse, partially overlapping DG context ensembles.

    A fraction `frac` of CA3 columns is gated ON (gate 1) per context and the
    rest nearly OFF (gate `lo`). The two ensembles are anti-correlated with
    overlap `2*frac - 1`, so distinct contexts engage distinct ensembles
    (remapping) while the shared core keeps that remapping *partial*.
    """
    u = rng.random(n)
    return np.vstack([np.where(u < frac, 1.0, lo),
                      np.where(u > 1 - frac, 1.0, lo)])


def bump_inputs(n: int, sigma: float = 1.6) -> np.ndarray:
    """Row-normalized Gaussian EC input bumps, one per state."""
    x = np.arange(n)
    X = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    return X / X.sum(1, keepdims=True)


def flat_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation between two rate maps, flattened."""
    a, b = A.ravel() - A.mean(), B.ravel() - B.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
