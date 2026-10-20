"""Index vs. single-stock vol: implied correlation and skew.

For an index I = sum_i w_i X_i, variance is
    sigma_I^2 = sum_i w_i^2 sigma_i^2 + sum_{i != j} w_i w_j rho_ij sigma_i sigma_j.
Setting every rho_ij to one number rho and solving gives the implied correlation
    rho = (sigma_I^2 - sum w_i^2 sigma_i^2) / ((sum w_i sigma_i)^2 - sum w_i^2 sigma_i^2).
This is how CBOE's implied-correlation indices work (top-50 names, normalised weights);
here the basket is 20 names, so treat the level as illustrative.

Index downside skew is steeper than single-stock skew because correlations rise in
sell-offs: index puts pay off when everything falls together, and the persistent demand
for index puts as portfolio protection (the "forced flows" story) is what that skew charges for.
Applying the formula to 25-delta put vols vs. ATM vols shows it as a higher
implied correlation on the downside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_correlation(index_vol: float, vols: np.ndarray, weights: np.ndarray) -> float:
    vols, w = np.asarray(vols, float), np.asarray(weights, float)
    w = w / w.sum()
    own = np.sum(w**2 * vols**2)
    return float((index_vol**2 - own) / (np.sum(w * vols) ** 2 - own))


def realized_correlation(returns: pd.DataFrame, index_returns: pd.Series, weights: pd.Series) -> dict:
    """Same formula with realized vols, plus the direct cap-weighted average pairwise correlation."""
    ann = np.sqrt(252)
    vols = returns.std() * ann
    w = weights.reindex(returns.columns)
    w = w / w.sum()
    formula = implied_correlation(index_returns.std() * ann, vols.to_numpy(), w.to_numpy())
    C = returns.corr().to_numpy()
    W = np.outer(w, w)
    np.fill_diagonal(W, 0.0)
    pairwise = float((W * C).sum() / W.sum())
    return {"formula": formula, "pairwise_cap_weighted": pairwise}
