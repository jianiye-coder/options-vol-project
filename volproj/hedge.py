"""Delta-hedged long option: P&L attribution into gamma, theta and carry.

Setup per window: on day 0 buy one ATM call at implied vol sigma_imp, then each close
re-hedge to -delta shares, with delta computed at sigma_imp ("hedge at implied"). The
option is also marked at sigma_imp until expiry, where it is worth its payoff. Cash
earns/pays r; the short stock pays actual cash dividends on ex-dates.

Dividends use the escrowed-dividend model: the option is priced on
S_adj = S - PV(cash dividends going ex before expiry), which is continuous through an
ex-date. Pricing with a continuous trailing yield while paying lumpy cash dividends on
the hedge would move ~13% of premium of P&L into whichever windows contain an ex-date.

Daily P&L (step i -> i+1, dt = 1/252), actual cash flows:
    total    = dV - delta_i*dS + r*(delta_i*S_i - V_i)*dt - delta_i*D_{i+1}
Attribution (Taylor expansion of V in S_adj and t):
    gamma    = 0.5 * gamma_i * dS_adj^2        (convexity: earned from movement)
    theta    = theta_i * dt                     (BS time decay at sigma_imp)
    carry    = financing + dividends on the hedge + delta*(dS_adj - dS)
             ~= r*(delta_i*S_adj_i - V_i)*dt
    residual = total - gamma - theta - carry    (higher-order terms, discrete hedging)

The Black–Scholes PDE gives theta + carry ~= -0.5 * sigma_imp^2 * S_adj^2 * gamma * dt, so
    total ~= sum_i 0.5 * gamma_i * S_adj_i^2 * (ret_i^2 - sigma_imp^2 * dt)
i.e. each day you win if the squared return beat the variance implied by the option
price. That is why a delta-hedged option is a bet on realized vs. implied vol, not on
direction.

Implied vol for the backtest is VIX/100 at entry (free history going back decades).
VIX is a 30-day variance-swap-style index on SPX that includes the put wing, so it sits
above SPY ATM implied vol; the summary reports how large that gap was on the snapshot
date and how much it changes the result. The window's actual dividends are treated as
known at entry: SPY's quarterly ex-dates are scheduled and amounts are predictable, so
this is a small, deliberate simplification rather than a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs
from .data import irx_to_cc

TRADING_DAYS = 252


def simulate_window(S, sigma_imp, r, q=0.0, dividends=None, strike=None, is_call=True,
                    dt=1.0 / TRADING_DAYS) -> pd.DataFrame:
    """S: closes for days 0..n. dividends[j]: cash dividend going ex on day j (day 0 ignored).
    q: optional continuous yield on top (used by tests; the backtest passes cash dividends)."""
    S = np.asarray(S, dtype=float)
    n = len(S) - 1
    D = np.zeros(n + 1) if dividends is None else np.asarray(dividends, dtype=float).copy()
    D[0] = 0.0
    steps = np.arange(n + 1)
    # PV at day i of dividends going ex strictly after i (and by expiry).
    PV = np.array([np.sum(D[i + 1:] * np.exp(-r * (steps[i + 1:] - i) * dt)) for i in range(n + 1)])
    S_adj = S - PV
    K = S[0] if strike is None else strike
    tau = (n - steps) * dt

    V = np.empty(n + 1)
    V[:n] = bs.price(S_adj[:n], K, tau[:n], r, q, sigma_imp, is_call)
    V[n] = max(S[n] - K, 0.0) if is_call else max(K - S[n], 0.0)
    g = bs.greeks(S_adj[:n], K, tau[:n], r, q, sigma_imp, is_call)
    delta = g["delta"]

    dS, dS_adj, dV = np.diff(S), np.diff(S_adj), np.diff(V)
    financing = r * (delta * S[:n] - V[:n]) * dt
    divs = -delta * (D[1:] + q * S[:n] * dt)
    total = dV - delta * dS + financing + divs
    gamma_pnl = 0.5 * g["gamma"] * dS_adj**2
    theta_pnl = g["theta"] * dt
    carry_pnl = financing + divs + delta * (dS_adj - dS)
    ret = np.log(S_adj[1:] / S_adj[:n])
    return pd.DataFrame({
        "S": S[:n], "S_adj": S_adj[:n], "tau": tau[:n], "V": V[:n], "delta": delta,
        "gamma": g["gamma"], "vega": g["vega"], "ret": ret,
        "total": total, "gamma_pnl": gamma_pnl, "theta_pnl": theta_pnl, "carry_pnl": carry_pnl,
        "residual": total - gamma_pnl - theta_pnl - carry_pnl,
        "approx": 0.5 * g["gamma"] * S_adj[:n] ** 2 * (ret**2 - sigma_imp**2 * dt),
    })


def window_from_history(h: pd.DataFrame, i: int, horizon: int = 21, vol_shift: float = 0.0):
    """Simulate the window starting at row i of the SPY/VIX/IRX history. Returns (sim, sigma, r)."""
    w = h.iloc[i:i + horizon + 1]
    sigma = w["vix"].iloc[0] / 100.0 + vol_shift
    r = irx_to_cc(w["irx"].iloc[0])
    sim = simulate_window(w["spy_close"].to_numpy(), sigma, r, dividends=w["spy_dividend"].to_numpy())
    return sim, sigma, r


def rolling_backtest(hist: pd.DataFrame, horizon: int = 21, vol_shift: float = 0.0,
                     start: str = "2011-01-01") -> pd.DataFrame:
    """Non-overlapping windows of ``horizon`` trading days; one ATM call per window.

    vol_shift is added to VIX/100 to test sensitivity (e.g. -0.03 = 3 vol points cheaper).
    """
    h = hist.dropna(subset=["spy_close", "vix", "irx"])
    first = h.index.searchsorted(pd.Timestamp(start))
    rows = []
    for i in range(first, len(h) - horizon, horizon):
        sim, sigma, r = window_from_history(h, i, horizon, vol_shift)
        vega0 = sim["vega"].iloc[0] / 100.0  # $ per vol point at entry
        realized = float(np.sqrt(TRADING_DAYS / horizon * (sim["ret"] ** 2).sum()))
        rows.append({
            "entry": h.index[i].date(), "exit": h.index[i + horizon].date(), "S0": sim["S"].iloc[0],
            "implied": sigma, "realized": realized, "premium": sim["V"].iloc[0],
            "has_ex_div": bool(h["spy_dividend"].iloc[i + 1:i + horizon + 1].gt(0).any()),
            "total": sim["total"].sum(), "gamma_pnl": sim["gamma_pnl"].sum(),
            "theta_pnl": sim["theta_pnl"].sum(), "carry_pnl": sim["carry_pnl"].sum(),
            "residual": sim["residual"].sum(), "approx": sim["approx"].sum(),
            "vega0": vega0, "pnl_vol_pts": sim["total"].sum() / vega0,
        })
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> dict:
    spread = (bt["implied"] - bt["realized"]) * 100
    pnl_pct = bt["total"] / bt["premium"] * 100
    fit = np.polyfit((bt["realized"] - bt["implied"]) * 100, bt["pnl_vol_pts"], 1)
    ex = bt["has_ex_div"]
    return {
        "n_windows": int(len(bt)),
        "first_entry": str(bt["entry"].iloc[0]), "last_exit": str(bt["exit"].iloc[-1]),
        "mean_implied_minus_realized_pts": float(spread.mean()),
        "median_implied_minus_realized_pts": float(spread.median()),
        "n_windows_realized_above_implied": int((bt["realized"] > bt["implied"]).sum()),
        "share_windows_realized_above_implied": float((bt["realized"] > bt["implied"]).mean()),
        "share_windows_pnl_positive": float((bt["total"] > 0).mean()),
        "mean_pnl_pct_premium": float(pnl_pct.mean()),
        "median_pnl_pct_premium": float(pnl_pct.median()),
        "best_pnl_pct_premium": float(pnl_pct.max()),
        "best_pnl_entry": str(bt.loc[pnl_pct.idxmax(), "entry"]),
        "worst_pnl_pct_premium": float(pnl_pct.min()),
        "years_with_positive_mean": int((pnl_pct.groupby(pd.to_datetime(bt["entry"]).dt.year).mean() > 0).sum()),
        "n_years": int(pd.to_datetime(bt["entry"]).dt.year.nunique()),
        "mean_pnl_pct_premium_ex_div_windows": float(pnl_pct[ex].mean()),
        "mean_pnl_pct_premium_no_div_windows": float(pnl_pct[~ex].mean()),
        "corr_pnl_vs_vol_spread": float(np.corrcoef(bt["realized"] - bt["implied"], bt["pnl_vol_pts"])[0, 1]),
        "slope_pnl_vol_pts_per_vol_pt": float(fit[0]),
        "mean_abs_residual_pct_premium": float((bt["residual"].abs() / bt["premium"]).mean() * 100),
    }
