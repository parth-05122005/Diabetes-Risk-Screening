"""Model-agnostic explanations via Monte-Carlo Shapley values.

For a patient x, each feature's Shapley value is its average marginal
contribution to the predicted risk, taken over random feature orderings and
random reference patients drawn from the training data (interventional
Shapley, Strumbelj & Kononenko 2014). The values are additive:

    sum(phi) ~= f(x) - E[f(background)]

so the explanation reads as "baseline risk + contributions = this patient's
risk". Works for any model exposing ``predict_proba``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def shapley_values(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Return (phi, baseline) where baseline = E[f(background)]."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    # Antithetic sampling: each permutation is paired with its reverse to cut variance.
    half = max(1, n_permutations // 2)
    perms = [rng.permutation(d) for _ in range(half)]
    perms += [p[::-1] for p in perms]

    rows = np.empty((len(perms), d + 1, d))
    for i, perm in enumerate(perms):
        z = background[rng.integers(len(background))]
        path = np.tile(z, (d + 1, 1))
        # Row k has the first k features of the permutation switched to the patient's values.
        for k, j in enumerate(perm):
            path[k + 1:, j] = x[j]
        rows[i] = path

    preds = predict(rows.reshape(-1, d)).reshape(len(perms), d + 1)
    deltas = np.diff(preds, axis=1)
    phi = np.zeros(d)
    for i, perm in enumerate(perms):
        phi[perm] += deltas[i]
    phi /= len(perms)
    baseline = float(predict(background).mean())
    return phi, baseline


def global_importance(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 32,
) -> np.ndarray:
    """Mean |Shapley value| over a set of patients."""
    phis = [shapley_values(predict, row, background, n_permutations, seed=i)[0] for i, row in enumerate(X)]
    return np.abs(np.array(phis)).mean(axis=0)
