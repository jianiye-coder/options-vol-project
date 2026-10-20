"""Implied volatility by inverting Black–Scholes.

Strategy:
1. Reject prices outside the no-arbitrage band [intrinsic on the forward, discounted
   upper bound]; no volatility can reproduce them, so we return NaN rather than a
   fake number.
2. Brent's method on a bracket [1e-4, 5], doubling the upper end (up to 20) if needed.
   BS price is strictly increasing in sigma, so a sign change guarantees a unique root.
3. If Brent cannot run (no bracket, non-finite values), fall back to a safeguarded
   Newton iteration on vega started from the Brenner–Subrahmanyam ATM approximation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from . import bs

SIGMA_LO = 1e-4
SIGMA_HI = 5.0
SIGMA_CAP = 20.0


def no_arb_bounds(S, K, T, r, q, is_call) -> tuple[float, float]:
    fwd_s = S * math.exp(-q * T)
    pv_k = K * math.exp(-r * T)
    if is_call:
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def _newton(target, S, K, T, r, q, is_call, tol=1e-10, max_iter=100) -> float:
    sigma = math.sqrt(2.0 * math.pi / T) * target / S  # Brenner–Subrahmanyam guess
    sigma = min(max(sigma, 0.05), 3.0)
    for _ in range(max_iter):
        diff = bs.price(S, K, T, r, q, sigma, is_call) - target
        if abs(diff) < tol:
            return sigma
        v = bs.vega(S, K, T, r, q, sigma)
        if not np.isfinite(v) or v < 1e-12:
            return float("nan")
        step = diff / v
        # Safeguard: never move more than halfway to zero or by more than 1.0 vol.
        step = max(min(step, 0.5 * sigma), -1.0)
        sigma -= step
        if not (SIGMA_LO <= sigma <= SIGMA_CAP):
            return float("nan")
    return float("nan")


def implied_vol(price, S, K, T, r, q, is_call, method: str = "auto", tol: float = 1e-10):
    """Return (sigma, method_used). sigma is NaN if no valid vol exists.

    method: "auto" (Brent, Newton fallback), "brent", or "newton".
    """
    if not (np.isfinite(price) and price > 0 and T > 0):
        return float("nan"), "invalid"
    lo_b, hi_b = no_arb_bounds(S, K, T, r, q, is_call)
    if price <= lo_b or price >= hi_b:
        return float("nan"), "no-arb"

    if method == "newton":
        return _newton(price, S, K, T, r, q, is_call, tol=tol), "newton"

    def f(s):
        return bs.price(S, K, T, r, q, s, is_call) - price

    hi = SIGMA_HI
    try:
        f_lo = f(SIGMA_LO)
        if f_lo > 0:  # price is below what even a ~0 vol implies (rounding at the bound)
            return float("nan"), "below-min-vol"
        f_hi = f(hi)
        while f_hi < 0 and hi < SIGMA_CAP:
            hi *= 2.0
            f_hi = f(hi)
        sigma = brentq(f, SIGMA_LO, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200)
        return float(sigma), "brent"
    except (ValueError, RuntimeError):
        if method == "brent":
            return float("nan"), "brent-failed"
        return _newton(price, S, K, T, r, q, is_call, tol=tol), "newton"


def implied_vol_vec(prices, S, K, T, r, q, is_call):
    """Loop wrapper over arrays; returns (sigmas, methods)."""
    prices, S, K, T, r, q, is_call = np.broadcast_arrays(
        np.asarray(prices, float), np.asarray(S, float), np.asarray(K, float),
        np.asarray(T, float), np.asarray(r, float), np.asarray(q, float), np.asarray(is_call, bool),
    )
    out = np.full(prices.shape, np.nan)
    methods = np.empty(prices.shape, dtype=object)
    for i in np.ndindex(prices.shape):
        out[i], methods[i] = implied_vol(prices[i], S[i], K[i], T[i], r[i], q[i], bool(is_call[i]))
    return out, methods
