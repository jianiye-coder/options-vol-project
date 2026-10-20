"""Earnings: the move the option market implies vs. the moves the stock actually made.

Implied move. Let w(T) = sigma_atm(T)^2 * T be ATM total variance. Take the last expiry
before the announcement (T1) and the first one after it (T2). Assuming the non-event
("diffusive") vol over [T1, T2] equals the pre-event vol sigma1, the extra variance the
market charges for the event is

    event_var = w(T2) - sigma1^2 * T2

and the one-day event standard deviation is sqrt(event_var). For a normal move the
expected absolute move is sqrt(2/pi) * std; we report both. This is the standard "term
structure kink" method; it ignores the weekend/trading-day calendar, so it is a rough
estimate.

Realized moves are close-to-close log returns on the reaction day: the next trading day
for reports timestamped after noon ET (after-close), the same day for pre-open reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_event_move(summary: pd.DataFrame, event_date, after_close: bool = True) -> dict:
    """An expiry on the report date itself is pre-event for an after-close report and
    post-event for a pre-open one."""
    s = summary.dropna(subset=["atm_iv"]).sort_values("expiry")
    if after_close:
        before, after = s[s["expiry"] <= event_date], s[s["expiry"] > event_date]
    else:
        before, after = s[s["expiry"] < event_date], s[s["expiry"] >= event_date]
    if before.empty or after.empty:
        return {}
    b, a = before.iloc[-1], after.iloc[0]
    event_var = a["atm_iv"] ** 2 * a["T"] - b["atm_iv"] ** 2 * a["T"]
    if event_var <= 0:
        return {"expiry_before": b["expiry"], "expiry_after": a["expiry"], "event_std": np.nan}
    std = float(np.sqrt(event_var))
    return {
        "expiry_before": b["expiry"], "iv_before": float(b["atm_iv"]),
        "expiry_after": a["expiry"], "iv_after": float(a["atm_iv"]),
        "event_std": std, "implied_abs_move": float(np.sqrt(2 / np.pi) * std),
    }


def realized_earnings_moves(prices: pd.Series, earnings_ts: pd.Series, asof) -> pd.DataFrame:
    prices = prices.dropna()
    idx = prices.index
    rows = []
    for ts in earnings_ts:
        d = pd.Timestamp(ts.date())
        if d.date() >= asof:
            continue
        after_close = ts.hour >= 12
        pos = idx.searchsorted(d, side="right" if after_close else "left")
        if pos <= 0 or pos >= len(idx):
            continue
        react, prev = idx[pos], idx[pos - 1]
        rows.append({"earnings_ts": ts, "reaction_day": react.date(),
                     "move": float(np.log(prices[react] / prices[prev]))})
    return pd.DataFrame(rows)
