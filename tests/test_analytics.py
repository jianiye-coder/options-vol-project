"""Checks on the analytics built on top of the pricer: hedge attribution, correlation, surface."""
import numpy as np
import pandas as pd
import pytest

from volproj import bs
from volproj.corr import implied_correlation, realized_correlation
from volproj import iv as iv_mod
from volproj.events import implied_event_move, realized_earnings_moves
from volproj.hedge import simulate_window
from volproj.surface import build_surface, expiry_summary, interp_by_days, model_free_vol


def _gbm(n, sigma, seed, S0=100.0, dt=1 / 252):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return S0 * np.exp(np.cumsum(np.r_[0.0, -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z]))


def test_hedge_attribution_adds_up_and_residual_is_small():
    S = _gbm(21, 0.2, seed=1)
    sim = simulate_window(S, sigma_imp=0.2, r=0.03, q=0.01)
    parts = sim[["gamma_pnl", "theta_pnl", "carry_pnl", "residual"]].sum(axis=1)
    np.testing.assert_allclose(parts, sim["total"], atol=1e-12)
    assert abs(sim["residual"].sum()) < 0.1 * sim["V"].iloc[0]


def test_theta_plus_carry_equals_minus_half_sigma2_s2_gamma():
    # BS PDE identity behind the attribution, with dividends as a continuous yield.
    S, K, T, r, q, sig = 100.0, 100.0, 0.1, 0.04, 0.015, 0.25
    g = bs.greeks(S, K, T, r, q, sig, True)
    # carry on the hedge = financing r*(delta*S - V) minus dividends q*delta*S paid on the short
    lhs = g["theta"] + r * (g["delta"] * S - g["price"]) - q * g["delta"] * S
    assert lhs == pytest.approx(-0.5 * sig**2 * S**2 * g["gamma"], rel=1e-10)


def test_hedged_pnl_sign_tracks_realized_vs_implied():
    # Average over many paths: realized 30% vs implied 20% should make money, and vice versa.
    win = np.mean([simulate_window(_gbm(21, 0.30, s), 0.20, 0.0, 0.0)["total"].sum() for s in range(200)])
    lose = np.mean([simulate_window(_gbm(21, 0.10, s), 0.20, 0.0, 0.0)["total"].sum() for s in range(200)])
    flat = np.mean([simulate_window(_gbm(21, 0.20, s), 0.20, 0.0, 0.0)["total"].sum() for s in range(400)])
    assert win > 0 > lose
    assert abs(flat) < 0.2 * abs(lose)


def test_implied_correlation_recovers_input():
    w = np.array([0.5, 0.3, 0.2])
    vols = np.array([0.30, 0.25, 0.40])
    rho = 0.45
    cov = rho * np.outer(vols, vols)
    np.fill_diagonal(cov, vols**2)
    index_vol = np.sqrt(w @ cov @ w)
    assert implied_correlation(index_vol, vols, w) == pytest.approx(rho)


def test_realized_correlation_on_synthetic_returns():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(5000)
    rets = pd.DataFrame({c: 0.6 * m + 0.8 * rng.standard_normal(5000) for c in "ABCD"}) * 0.01
    w = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4}, dtype=float)
    idx = (rets * (w / w.sum())).sum(axis=1)
    out = realized_correlation(rets, idx, w)
    assert out["pairwise_cap_weighted"] == pytest.approx(0.36, abs=0.03)
    assert out["formula"] == pytest.approx(out["pairwise_cap_weighted"], abs=0.03)


def _synthetic_chain(S, r, q, vol_fn, expiries, asof, k_range=(0.7, 1.3)):
    rows = []
    for exp in expiries:
        T = (pd.Timestamp(exp) - pd.Timestamp(asof)).days / 365
        for K in np.arange(k_range[0] * S, k_range[1] * S, 1.0):
            for is_call in (True, False):
                F = S * np.exp((r - q) * T)
                p = bs.price(S, K, T, r, q, vol_fn(np.log(K / F), T), is_call)
                rows.append({"strike": K, "bid": p * 0.99, "ask": p * 1.01, "type": "call" if is_call else "put",
                             "expiry": pd.Timestamp(exp).date(), "volume": 1, "openInterest": 1,
                             "impliedVolatility": np.nan})
    return pd.DataFrame(rows)


def test_surface_recovers_known_smile_and_forward():
    S, r, q, asof = 200.0, 0.04, 0.012, pd.Timestamp("2026-01-02").date()
    vol_fn = lambda k, T: 0.22 - 0.25 * k + 0.6 * k**2
    raw = _synthetic_chain(S, r, q, vol_fn, ["2026-02-20", "2026-04-17"], asof)
    # Feed a wrong dividend yield: the parity forward should still recover the true carry.
    surf = build_surface(raw, asof, S, r, q=0.05, forward="parity")
    true_iv = vol_fn(np.log(surf["strike"] / (S * np.exp((r - q) * surf["T"]))), surf["T"])
    np.testing.assert_allclose(surf["iv"], true_iv, atol=2e-3)  # 1% bid/ask noise on mids is symmetric
    np.testing.assert_allclose(surf["q"], q, atol=1e-4)
    es = expiry_summary(surf)
    assert (es["atm_iv"] - 0.22).abs().max() < 2e-3
    assert (es["put_skew25"] > 0).all()


def test_implied_event_move():
    summ = pd.DataFrame({
        "expiry": pd.to_datetime(["2026-01-09", "2026-01-16"]).date,
        "T": [7 / 365, 14 / 365], "atm_iv": [0.25, np.sqrt((0.25**2 * 14 / 365 + 0.05**2) / (14 / 365))],
    })
    out = implied_event_move(summ, pd.Timestamp("2026-01-12").date())
    assert out["event_std"] == pytest.approx(0.05, rel=1e-9)


def test_hedge_cash_flows_on_a_deterministic_path_net_to_zero():
    # Deep ITM call (delta ~ 1, gamma ~ 0), the dividend-adjusted price grows at exactly r and
    # one cash dividend goes ex mid-window. Being long the call and short delta shares is then
    # a riskless position, so hedged P&L must be ~0. A sign error in the financing line
    # (~r*S*T = 0.4) or the dividend line (~2*D = 2) would show up immediately.
    n, dt, r, D, ex_day = 21, 1 / 252, 0.05, 1.0, 10
    steps = np.arange(n + 1)
    divs = np.zeros(n + 1)
    divs[ex_day] = D
    pv = np.array([np.sum(divs[i + 1:] * np.exp(-r * (steps[i + 1:] - i) * dt)) for i in range(n + 1)])
    S = 100.0 * np.exp(r * steps * dt) + pv
    sim = simulate_window(S, sigma_imp=0.05, r=r, dividends=divs, strike=50.0)
    assert abs(sim["total"].sum()) < 1e-3
    assert abs(sim["gamma_pnl"].sum()) < 1e-6
    # carry ~= r * (delta * S_adj - V) * dt, and theta + carry ~= 0 when gamma ~ 0
    np.testing.assert_allclose(sim["carry_pnl"], r * (sim["delta"] * sim["S_adj"] - sim["V"]) * dt, atol=1e-4)
    assert abs((sim["theta_pnl"] + sim["carry_pnl"]).sum()) < 1e-3


def test_hedged_pnl_is_unbiased_with_rates_and_dividends():
    # Realized = implied with r > 0 and a cash dividend: mean hedged P&L ~ 0 (no carry leak).
    n, dt, r, D = 21, 1 / 252, 0.05, 0.8
    steps = np.arange(n + 1)
    divs = np.zeros(n + 1)
    divs[8] = D
    pv = np.array([np.sum(divs[i + 1:] * np.exp(-r * (steps[i + 1:] - i) * dt)) for i in range(n + 1)])
    pnl = []
    for s in range(400):
        rng = np.random.default_rng(s)
        z = rng.standard_normal(n)
        S_adj = 100 * np.exp(np.cumsum(np.r_[0.0, (r - 0.02) * dt + 0.2 * np.sqrt(dt) * z]))
        sim = simulate_window(S_adj + pv, 0.2, r, dividends=divs)
        pnl.append(sim["total"].sum() / sim["V"].iloc[0])
    assert abs(np.mean(pnl)) < 0.03  # within ~3% of premium; std error is ~1%


def test_model_free_vol_flat_smile():
    S, r, asof = 100.0, 0.03, pd.Timestamp("2026-01-02").date()
    raw = _synthetic_chain(S, r, 0.0, lambda k, T: 0.20, ["2026-02-02"], asof, k_range=(0.3, 2.0))
    raw.loc[raw["bid"] < 1e-4, ["bid", "ask"]] = 0.0  # zero bids in the far wings, as in real chains
    expiry = raw["expiry"].iloc[0]
    assert model_free_vol(raw, asof, S, r, expiry) == pytest.approx(0.20, abs=2e-3)


def test_25_delta_put_iv_value():
    S, r, q, asof = 200.0, 0.04, 0.0, pd.Timestamp("2026-01-02").date()
    vol_fn = lambda k, T: 0.22 - 0.25 * k + 0.6 * k**2
    raw = _synthetic_chain(S, r, q, vol_fn, ["2026-03-20"], asof)
    es = expiry_summary(build_surface(raw, asof, S, r, q))
    T = es["T"].iloc[0]
    F = S * np.exp(r * T)
    # Strike where the true put delta is -0.25, found on a fine grid.
    Ks = np.linspace(150, 200, 50001)
    d = bs.delta(S, Ks, T, r, q, vol_fn(np.log(Ks / F), T), False)
    K25 = Ks[np.abs(d + 0.25).argmin()]
    assert es["put25_iv"].iloc[0] == pytest.approx(vol_fn(np.log(K25 / F), T), abs=1e-3)
    assert es["put25_k"].iloc[0] == pytest.approx(np.log(K25 / F), abs=2e-3)


def test_interp_by_days_is_linear_in_total_variance():
    summ = pd.DataFrame({"T": [30 / 365, 90 / 365], "atm_iv": [0.20, 0.30]})
    w = 0.5 * (0.20**2 * 30 / 365 + 0.30**2 * 90 / 365)
    assert interp_by_days(summ, "atm_iv", 60) == pytest.approx(np.sqrt(w / (60 / 365)))
    assert np.isnan(interp_by_days(summ, "atm_iv", 120))  # no extrapolation


def test_realized_earnings_reaction_day():
    idx = pd.bdate_range("2026-01-05", periods=10)
    px = pd.Series(100.0 * np.exp(0.01 * np.arange(10)), index=idx)
    px.iloc[3] *= 1.05  # Thu 2026-01-08 close jumps 5%
    tz = "America/New_York"
    ts = pd.Series([pd.Timestamp("2026-01-07 16:05", tz=tz),   # after close Wed -> reacts Thu
                    pd.Timestamp("2026-01-08 07:00", tz=tz)])  # pre-open Thu -> reacts Thu
    out = realized_earnings_moves(px, ts, pd.Timestamp("2026-02-01").date())
    assert list(out["reaction_day"]) == [idx[3].date(), idx[3].date()]
    assert out["move"].iloc[0] == pytest.approx(0.01 + np.log(1.05))


def test_implied_event_move_expiry_on_event_day():
    summ = pd.DataFrame({"expiry": pd.to_datetime(["2026-01-09", "2026-01-16"]).date,
                         "T": [7 / 365, 14 / 365], "atm_iv": [0.25, 0.30]})
    day = pd.Timestamp("2026-01-09").date()
    assert implied_event_move(summ, day, after_close=True)["expiry_before"] == day
    assert implied_event_move(summ, day, after_close=False) == {}  # nothing strictly before


def test_auto_falls_back_to_newton_when_brent_fails(monkeypatch):
    def broken(*a, **k):
        raise ValueError("no bracket")
    monkeypatch.setattr(iv_mod, "brentq", broken)
    p = bs.price(100, 105, 0.5, 0.03, 0.01, 0.31, True)
    sigma, method = iv_mod.implied_vol(p, 100, 105, 0.5, 0.03, 0.01, True)
    assert method == "newton"
    assert sigma == pytest.approx(0.31, rel=1e-8)
    assert np.isnan(iv_mod.implied_vol(p, 100, 105, 0.5, 0.03, 0.01, True, method="brent")[0])
