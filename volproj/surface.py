"""Turn a raw Yahoo option chain into a cleaned implied-vol surface with Greeks.

Choices worth knowing about:
* Prices are bid/ask mids. Quotes with no bid, crossed markets, mids under $0.05 or
  relative spreads above ``max_rel_spread`` are dropped: their "IV" is mostly noise.
* Only out-of-the-money options are used (puts below the forward, calls above). US
  single-stock and SPY options are American; Black–Scholes ignores the early-exercise
  premium, which is negligible for OTM options but not for deep ITM ones.
* Rates: one flat T-bill rate. Dividends: two explicit options. ``forward="div"`` uses a
  continuous yield from trailing-12m cash dividends. ``forward="parity"`` (default) backs
  the forward out of near-the-money put-call parity for each expiry and converts it to an
  expiry-specific carry q. A trailing yield gets SPY's forward wrong in two ways: it
  spreads dividends evenly although they arrive quarterly (no ex-date before the Dec
  third Friday, one just after), and the 4:00pm stock close is slightly stale against
  the 4:15pm option close. A $1 forward error on SPY puts ~1 vol pt between same-strike
  call and put IVs, so the default is "parity", and ``same_strike_iv_gap`` reports the
  difference rather than hiding it.
* Time to expiry is calendar days / 365 measured from the snapshot close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bs
from .iv import implied_vol_vec


def prepare_chain(raw: pd.DataFrame, asof, spot: float, r: float, q: float) -> pd.DataFrame:
    df = raw.copy()
    df["T"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp(asof)).dt.days / 365.0
    df["is_call"] = df["type"].eq("call")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df["S"], df["r"], df["q_div"] = spot, r, q
    df["F_div"] = spot * np.exp((r - q) * df["T"])
    return df


def parity_forwards(df: pd.DataFrame, n_strikes: int = 5, max_rel_spread: float = 0.10) -> pd.DataFrame:
    """Per expiry: forward implied by put-call parity, F = K + e^{rT}(C - P).

    Uses the median over the ``n_strikes`` tightly-quoted strikes closest to the
    dividend-yield forward. Parity is exact only for European options; near the money
    the American early-exercise premium is tiny, so this is a good estimate of the
    forward the market is actually pricing (dividends, borrow, and any staleness between
    the 4:00pm stock close and the 4:15pm option close all end up in it).
    """
    rows = []
    cols = ["expiry", "T", "F_div", "F_parity", "gap_bp", "q_implied", "n_strikes", "spread_cap"]
    for exp, g in df.groupby("expiry"):
        # Tight quotes first; fall back to a looser spread cap for thinly quoted names.
        for cap in (max_rel_spread, 0.30):
            ok = g[(g["bid"] > 0) & (g["ask"] > g["bid"]) & (g["rel_spread"] <= cap)]
            piv = ok.pivot_table(index="strike", columns="type", values="mid")
            if {"call", "put"} <= set(piv.columns) and not piv.dropna().empty:
                piv = piv.dropna()
                break
        else:
            continue
        F_div, T, r, S = g["F_div"].iloc[0], g["T"].iloc[0], g["r"].iloc[0], g["S"].iloc[0]
        near = piv.iloc[np.abs(piv.index.to_numpy() - F_div).argsort()[:n_strikes]]
        F_par = float(np.median(near.index + np.exp(r * T) * (near["call"] - near["put"])))
        rows.append({"expiry": exp, "T": T, "F_div": F_div, "F_parity": F_par,
                     "gap_bp": 1e4 * (F_par / F_div - 1.0),
                     "q_implied": r - np.log(F_par / S) / T if T > 0 else np.nan,
                     "n_strikes": len(near), "spread_cap": cap})
    return pd.DataFrame(rows, columns=cols)


def _with_forward(df: pd.DataFrame, forward: str) -> pd.DataFrame:
    """Attach forward F, the carry q it implies, and log-moneyness k."""
    if forward == "parity":
        fw = parity_forwards(df).set_index("expiry")["F_parity"]
        df = df[df["expiry"].isin(fw.index)].copy()
        df["F"] = df["expiry"].map(fw)
    elif forward == "div":
        df = df.copy()
        df["F"] = df["F_div"]
    else:
        raise ValueError(forward)
    df["q"] = np.where(df["T"] > 0, df["r"] - np.log(df["F"] / df["S"]) / df["T"].where(df["T"] > 0), df["q_div"])
    df["k"] = np.log(df["strike"] / df["F"])
    return df


def same_strike_iv_gap(raw: pd.DataFrame, asof, spot: float, r: float, q: float, forward: str,
                       min_days: int = 7, max_rel_spread: float = 0.30) -> float:
    """Median over expiries of |IV(call) - IV(put)| (vol points) at the quoted strike nearest F.

    For European options parity forces these to be equal, so the gap measures how far
    the forward assumption is from the one the market is pricing.
    """
    df = _with_forward(prepare_chain(raw, asof, spot, r, q), forward)
    df = df[(df["T"] >= min_days / 365.0) & (df["bid"] > 0) & (df["ask"] > df["bid"])
            & (df["rel_spread"] <= max_rel_spread)]
    gaps = []
    for _, g in df.groupby("expiry"):
        both = g.groupby("strike")["type"].nunique()
        both = both[both == 2].index.to_numpy()
        if len(both) == 0:
            continue
        K = both[np.abs(both - g["F"].iloc[0]).argmin()]
        row = g[g["strike"] == K].set_index("type")
        iv, _ = implied_vol_vec(row["mid"], row["S"], K, row["T"], row["r"], row["q"], row.index == "call")
        gaps.append(abs(iv[list(row.index).index("call")] - iv[list(row.index).index("put")]))
    return float(np.nanmedian(gaps) * 100) if gaps else float("nan")


def build_surface(
    raw: pd.DataFrame,
    asof,
    spot: float,
    r: float,
    q: float,
    min_days: int = 7,
    max_rel_spread: float = 0.30,
    min_mid: float = 0.05,
    max_abs_k: float = 0.5,
    forward: str = "parity",
) -> pd.DataFrame:
    """forward="parity" (default) prices off the put-call-parity forward per expiry by
    setting an expiry-specific carry q = r - ln(F/S)/T; forward="div" uses the trailing
    dividend yield for every expiry."""
    df = _with_forward(prepare_chain(raw, asof, spot, r, q), forward)
    otm = (df["is_call"] & (df["strike"] >= df["F"])) | (~df["is_call"] & (df["strike"] < df["F"]))
    keep = (
        otm
        & (df["T"] >= min_days / 365.0)
        & (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["mid"] >= min_mid)
        & (df["rel_spread"] <= max_rel_spread)
        & (df["k"].abs() <= max_abs_k)
    )
    df = df[keep].copy()

    iv, method = implied_vol_vec(df["mid"], df["S"], df["strike"], df["T"], df["r"], df["q"], df["is_call"])
    df["iv"], df["iv_method"] = iv, method
    df = df[np.isfinite(df["iv"]) & (df["iv"] > 0.01) & (df["iv"] < 3.0)].copy()

    g = bs.scale_greeks(bs.greeks(df["S"].to_numpy(), df["strike"].to_numpy(), df["T"].to_numpy(),
                                  df["r"].to_numpy(), df["q"].to_numpy(), df["iv"].to_numpy(),
                                  df["is_call"].to_numpy()))
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        df[name] = g[name]
    df["model_minus_mid"] = g["price"] - df["mid"]  # should be ~0: checks the inversion
    cols = ["expiry", "T", "type", "strike", "k", "bid", "ask", "mid", "rel_spread", "volume",
            "openInterest", "iv", "iv_method", "impliedVolatility", "delta", "gamma", "vega",
            "theta", "rho", "model_minus_mid", "S", "F", "r", "q"]
    return df[cols].rename(columns={"impliedVolatility": "yahoo_iv"}).sort_values(["expiry", "strike"])


def _interp_at_k0(g: pd.DataFrame) -> float:
    g = g.sort_values("k")
    below, above = g[g["k"] < 0], g[g["k"] >= 0]
    if below.empty or above.empty:
        return np.nan
    a, b = below.iloc[-1], above.iloc[0]
    w = (0 - a["k"]) / (b["k"] - a["k"])
    return float(a["iv"] + w * (b["iv"] - a["iv"]))


def _at_delta(g: pd.DataFrame, target: float, col: str = "iv") -> float:
    g = g.sort_values("delta")
    d, v = g["delta"].to_numpy(), g[col].to_numpy()
    if len(d) < 2 or not (d.min() <= target <= d.max()):
        return np.nan
    return float(np.interp(target, d, v))


def expiry_summary(surf: pd.DataFrame) -> pd.DataFrame:
    """ATM vol and 25-delta wings per expiry (the term structure and skew in one table)."""
    rows = []
    for exp, g in surf.groupby("expiry"):
        puts, calls = g[g["type"] == "put"], g[g["type"] == "call"]
        atm = _interp_at_k0(g)
        p25, c25 = _at_delta(puts, -0.25), _at_delta(calls, 0.25)
        rows.append({
            "expiry": exp, "T": g["T"].iloc[0], "days": int(round(g["T"].iloc[0] * 365)),
            "n_quotes": len(g), "atm_iv": atm, "put25_iv": p25, "call25_iv": c25,
            "rr25": p25 - c25, "put_skew25": p25 - atm, "put25_k": _at_delta(puts, -0.25, "k"),
        })
    return pd.DataFrame(rows)


def interp_by_days(summary: pd.DataFrame, col: str, days: float) -> float:
    """Interpolate a summary column in total-variance space for ATM-like vol columns."""
    s = summary.dropna(subset=[col]).sort_values("T")
    T = days / 365.0
    if s.empty or not (s["T"].min() <= T <= s["T"].max()):
        return np.nan
    w = np.interp(T, s["T"], s[col] ** 2 * s["T"])
    return float(np.sqrt(w / T))


def model_free_vol(raw: pd.DataFrame, asof, spot: float, r: float, expiry) -> float:
    """CBOE VIX-style model-free implied vol for one expiry (no wing truncation rules).

    sigma^2 = (2/T) * sum(dK/K^2 * e^{rT} * Q(K)) - (1/T) * (F/K0 - 1)^2, with Q the OTM
    mid (put below K0, call above, average at K0), F the parity forward and K0 the first
    strike at or below F. Comparing it with ATM IV shows how much of the VIX level comes
    from the put wing rather than from at-the-money vol.
    """
    df = prepare_chain(raw, asof, spot, r, 0.0)
    df = df[df["expiry"] == expiry]
    F = float(parity_forwards(df)["F_parity"].iloc[0])
    T = float(df["T"].iloc[0])
    quotes = df[df["bid"] > 0].pivot_table(index="strike", columns="type", values="mid")
    K0 = quotes.index[quotes.index <= F].max()
    q = pd.concat([quotes.loc[quotes.index < K0, "put"], quotes.loc[quotes.index > K0, "call"]])
    q.loc[K0] = quotes.loc[K0, ["put", "call"]].mean()
    q = q.dropna().sort_index()
    K = q.index.to_numpy(dtype=float)
    dK = np.gradient(K)
    var = 2.0 / T * np.sum(dK / K**2 * np.exp(r * T) * q.to_numpy()) - (F / K0 - 1.0) ** 2 / T
    return float(np.sqrt(var))
