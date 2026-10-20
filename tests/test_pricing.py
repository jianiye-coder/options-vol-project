import math

import numpy as np
import pytest

from volproj import bs
from volproj.iv import implied_vol, implied_vol_vec, no_arb_bounds

# A grid of strikes / maturities / vols / carry that covers ITM, ATM, OTM and short/long dated.
CASES = [
    (S, K, T, r, q, sigma)
    for S in (100.0,)
    for K in (60.0, 90.0, 100.0, 110.0, 160.0)
    for T in (7 / 365, 0.25, 1.0, 3.0)
    for r in (0.0, 0.045)
    for q in (0.0, 0.02)
    for sigma in (0.08, 0.25, 0.9)
]


def test_known_value():
    # Textbook example (Hull, Options, Futures & Other Derivatives): S=42, K=40, r=10%, sigma=20%, T=0.5.
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, True) == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, False) == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_put_call_parity(S, K, T, r, q, sigma):
    c = bs.price(S, K, T, r, q, sigma, True)
    p = bs.price(S, K, T, r, q, sigma, False)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(S, K, T, r, q, sigma, is_call):
    g = bs.greeks(S, K, T, r, q, sigma, is_call)
    P = lambda **kw: bs.price(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call), **kw})

    hS, hv, hr = 1e-4 * S, 1e-5, 1e-6
    hT = min(1e-5, T / 10)
    fd = {
        "delta": (P(S=S + hS) - P(S=S - hS)) / (2 * hS),
        "gamma": (P(S=S + hS) - 2 * g["price"] + P(S=S - hS)) / hS**2,
        "vega": (P(sigma=sigma + hv) - P(sigma=sigma - hv)) / (2 * hv),
        "theta": -(P(T=T + hT) - P(T=T - hT)) / (2 * hT),
        "rho": (P(r=r + hr) - P(r=r - hr)) / (2 * hr),
    }
    for name, approx in fd.items():
        # Relative tolerance with an absolute floor for Greeks that are ~0 deep OTM/ITM.
        assert g[name] == pytest.approx(approx, rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
@pytest.mark.parametrize("is_call", [True, False])
def test_iv_roundtrip(S, K, T, r, q, sigma, is_call):
    p = bs.price(S, K, T, r, q, sigma, is_call)
    lo, hi = no_arb_bounds(S, K, T, r, q, is_call)
    # Skip prices that are numerically indistinguishable from the no-arb bound: vol is not
    # identifiable there (vega ~ 0), which is exactly why the surface code drops them.
    if p - lo < 1e-8 * S or hi - p < 1e-8 * S:
        pytest.skip("price at no-arbitrage bound; vol not identifiable")
    iv, method = implied_vol(p, S, K, T, r, q, is_call)
    assert method == "brent"
    assert iv == pytest.approx(sigma, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("K,T,sigma", [(100, 0.25, 0.2), (90, 1.0, 0.35), (120, 0.5, 0.6), (100, 2.0, 1.2)])
def test_newton_fallback_agrees_with_brent(K, T, sigma):
    S, r, q = 100.0, 0.04, 0.01
    for is_call in (True, False):
        p = bs.price(S, K, T, r, q, sigma, is_call)
        iv_n, m = implied_vol(p, S, K, T, r, q, is_call, method="newton")
        assert m == "newton"
        assert iv_n == pytest.approx(sigma, rel=1e-7)


def test_rejects_arbitrage_violations():
    S, K, T, r, q = 100.0, 90.0, 0.5, 0.04, 0.0
    lo, hi = no_arb_bounds(S, K, T, r, q, True)
    assert math.isnan(implied_vol(lo * 0.99, S, K, T, r, q, True)[0])  # below intrinsic
    assert math.isnan(implied_vol(hi * 1.01, S, K, T, r, q, True)[0])  # above spot
    assert math.isnan(implied_vol(-1.0, S, K, T, r, q, True)[0])


def test_vectorised_matches_scalar():
    K = np.array([80.0, 100.0, 120.0])
    sig = np.array([0.3, 0.2, 0.25])
    p = bs.price(100.0, K, 0.5, 0.03, 0.01, sig, False)
    iv, methods = implied_vol_vec(p, 100.0, K, 0.5, 0.03, 0.01, False)
    np.testing.assert_allclose(iv, sig, rtol=1e-7)
    assert set(methods) == {"brent"}


def test_scale_greeks_units():
    g = bs.greeks(100, 100, 1.0, 0.0, 0.0, 0.2, True)
    s = bs.scale_greeks(g)
    assert s["vega"] == pytest.approx(g["vega"] / 100)
    assert s["theta"] == pytest.approx(g["theta"] / 365)
