"""Black–Scholes–Merton prices and Greeks for European options with a continuous dividend yield.

Conventions (all inputs annualised, continuously compounded):
    S spot, K strike, T years to expiry, r risk-free rate, q dividend yield, sigma vol.
    ``is_call`` is a bool (or bool array): True for calls, False for puts.

Greeks are returned as raw partial derivatives:
    delta = dV/dS, gamma = d2V/dS2, vega = dV/dsigma (per 1.00 of vol),
    theta = dV/dt = -dV/dT (per year), rho = dV/dr (per 1.00 of rate).
Use ``scale_greeks`` for desk units (vega per vol point, theta per calendar day, rho per 1%).
All functions broadcast over numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, fast and vectorised

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _scalarize(x):
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, r, q, sigma))
    vsqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vsqrt
    return d1, d1 - vsqrt


def price(S, K, T, r, q, sigma, is_call):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    fwd_s = S * np.exp(-q * T)  # spot net of dividends paid before expiry
    pv_k = K * np.exp(-r * T)
    call = fwd_s * ndtr(d1) - pv_k * ndtr(d2)
    put = pv_k * ndtr(-d2) - fwd_s * ndtr(-d1)
    return _scalarize(np.where(is_call, call, put))


def delta(S, K, T, r, q, sigma, is_call):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    dq = np.exp(-q * np.asarray(T, dtype=float))
    return _scalarize(np.where(is_call, dq * ndtr(d1), dq * (ndtr(d1) - 1.0)))


def gamma(S, K, T, r, q, sigma, is_call=True):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    S, T, q, sigma = (np.asarray(a, dtype=float) for a in (S, T, q, sigma))
    return _scalarize(np.exp(-q * T) * _npdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S, K, T, r, q, sigma, is_call=True):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    S, T, q = (np.asarray(a, dtype=float) for a in (S, T, q))
    return _scalarize(S * np.exp(-q * T) * _npdf(d1) * np.sqrt(T))


def theta(S, K, T, r, q, sigma, is_call):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    S, K, T, r, q, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, r, q, sigma))
    decay = -S * np.exp(-q * T) * _npdf(d1) * sigma / (2.0 * np.sqrt(T))
    call = decay - r * K * np.exp(-r * T) * ndtr(d2) + q * S * np.exp(-q * T) * ndtr(d1)
    put = decay + r * K * np.exp(-r * T) * ndtr(-d2) - q * S * np.exp(-q * T) * ndtr(-d1)
    return _scalarize(np.where(is_call, call, put))


def rho(S, K, T, r, q, sigma, is_call):
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    K, T, r = (np.asarray(a, dtype=float) for a in (K, T, r))
    call = K * T * np.exp(-r * T) * ndtr(d2)
    put = -K * T * np.exp(-r * T) * ndtr(-d2)
    return _scalarize(np.where(is_call, call, put))


def greeks(S, K, T, r, q, sigma, is_call) -> dict:
    args = (S, K, T, r, q, sigma, is_call)
    return {
        "price": price(*args),
        "delta": delta(*args),
        "gamma": gamma(*args),
        "vega": vega(*args),
        "theta": theta(*args),
        "rho": rho(*args),
    }


def scale_greeks(g: dict) -> dict:
    """Convert raw Greeks to desk units: vega per vol point, theta per calendar day, rho per 1%."""
    out = dict(g)
    out["vega"] = g["vega"] / 100.0
    out["theta"] = g["theta"] / 365.0
    out["rho"] = g["rho"] / 100.0
    return out
