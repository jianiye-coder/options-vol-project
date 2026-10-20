"""Download option chains and price history from Yahoo Finance and cache them to disk.

Everything downstream reads from ``data/raw/<snapshot-date>/``, so once a snapshot is
saved the analysis is reproducible without network access. Yahoo data is free and needs
no credentials; it is end-of-day-ish and occasionally stale, which is why the surface
code filters quotes aggressively.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Main surface: one index ETF and two liquid single names.
SURFACE_TICKERS = ["SPY", "AAPL", "NVDA"]

# Basket for the index-vs-single-stock comparison: 20 of the largest S&P 500 names.
# Weights are their market caps renormalised within the basket (a CBOE-style
# simplification; the real index has ~500 names).
BASKET_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
    "LLY", "V", "WMT", "ORCL", "XOM", "MA", "NFLX", "COST", "JNJ", "PG",
]

HISTORY_START = "2010-01-01"


def _yf():
    import yfinance as yf  # imported lazily so tests never touch the network

    return yf


def _fetch_chain(tk, expiries: list[str]) -> pd.DataFrame:
    frames = []
    for exp in expiries:
        for attempt in range(3):
            try:
                ch = tk.option_chain(exp)
                break
            except Exception:  # transient Yahoo errors
                if attempt == 2:
                    raise
                time.sleep(1.5)
        for kind, df in (("call", ch.calls), ("put", ch.puts)):
            df = df.copy()
            df["type"] = kind
            df["expiry"] = exp
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def snapshot(
    surface_tickers=SURFACE_TICKERS,
    basket_tickers=BASKET_TICKERS,
    surface_max_days: int = 400,
    basket_window: tuple[int, int] = (20, 75),
) -> Path:
    """Fetch chains, spots, trailing dividends, market caps and rates; save under a dated folder."""
    yf = _yf()
    spy_hist = yf.Ticker("SPY").history(period="5d", auto_adjust=False)
    asof = spy_hist.index[-1].date()
    out = RAW_DIR / asof.isoformat()
    (out / "chains").mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "asof": asof.isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance via yfinance",
        "tickers": {},
    }

    for sym in dict.fromkeys(list(surface_tickers) + list(basket_tickers)):
        tk = yf.Ticker(sym)
        hist = tk.history(period="5d", auto_adjust=False)
        spot = float(hist["Close"].iloc[-1])
        divs = tk.dividends
        if len(divs):
            cutoff = pd.Timestamp(asof, tz=divs.index.tz) - pd.Timedelta(days=365)
            div12m = float(divs[divs.index > cutoff].sum())
        else:
            div12m = 0.0
        try:
            mcap = tk.info.get("marketCap")
        except Exception:
            mcap = None

        lo, hi = (0, surface_max_days) if sym in surface_tickers else basket_window
        expiries = [
            e for e in tk.options
            if lo <= (date.fromisoformat(e) - asof).days <= hi
        ]
        chain = _fetch_chain(tk, expiries)
        chain.to_csv(out / "chains" / f"{sym}.csv", index=False)
        meta["tickers"][sym] = {
            "spot": spot,
            "dividends_trailing_12m": div12m,
            "market_cap": mcap,
            "n_expiries": len(expiries),
            "n_contracts": int(len(chain)),
        }
        print(f"  {sym:6s} spot={spot:9.2f}  expiries={len(expiries):3d}  contracts={len(chain):5d}")

    # Rates and vol index: 13-week T-bill discount yield (%) and VIX (%).
    for sym, key in (("^IRX", "irx_pct"), ("^VIX", "vix_pct")):
        meta[key] = float(yf.Ticker(sym).history(period="5d")["Close"].iloc[-1])

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    fetch_history(out, basket_tickers)
    fetch_event_data(out, [t for t in surface_tickers if t != "SPY"])
    return out


def fetch_event_data(out: Path, tickers, n_events: int = 16) -> None:
    """Past and next earnings timestamps plus split-adjusted closes, for the earnings-move study."""
    yf = _yf()
    rows = []
    for sym in tickers:
        ed = yf.Ticker(sym).get_earnings_dates(limit=n_events + 4)
        for ts, rec in ed.iterrows():
            rows.append({"ticker": sym, "earnings_ts": ts.isoformat(), "reported_eps": rec.get("Reported EPS")})
    pd.DataFrame(rows).to_csv(out / "earnings_dates.csv", index=False)
    px = yf.download(list(tickers), start="2019-01-01", end=(pd.Timestamp(out.name) + pd.Timedelta(days=1)).date().isoformat(),
                     auto_adjust=True, progress=False)["Close"]
    px.index.name = "date"
    px.to_csv(out / "history_surface_adjclose.csv")


def fetch_history(out: Path, basket_tickers=BASKET_TICKERS) -> None:
    """Daily history for the hedging backtest (SPY, VIX, IRX) and basket closes for realized correlation."""
    yf = _yf()
    end = (pd.Timestamp(json.loads((out / "meta.json").read_text())["asof"]) + pd.Timedelta(days=1)).date()
    syms = ["SPY", "^VIX", "^IRX"]
    px = yf.download(syms, start=HISTORY_START, end=end.isoformat(), auto_adjust=False, progress=False)
    hist = pd.DataFrame({
        "spy_close": px["Close"]["SPY"],
        "vix": px["Close"]["^VIX"],
        "irx": px["Close"]["^IRX"],
    }).dropna(subset=["spy_close"])
    spy_div = yf.Ticker("SPY").dividends
    spy_div.index = spy_div.index.tz_localize(None).normalize()
    hist["spy_dividend"] = spy_div.reindex(hist.index).fillna(0.0)
    hist.index.name = "date"
    hist.to_csv(out / "history_spy_vix_irx.csv")

    basket = yf.download(list(basket_tickers) + ["SPY"], period="1y", auto_adjust=True, progress=False)["Close"]
    basket.index.name = "date"
    basket.to_csv(out / "history_basket_adjclose.csv")


def irx_to_cc(irx_pct):
    """13-week T-bill discount yield (^IRX, percent) -> continuously compounded rate.

    Bill price P = 1 - d * 91/360, so r = -ln(P) * 365/91. One flat rate is used for every
    maturity; the curve slope is ignored (the parity forward absorbs most of the effect).
    """
    d = np.asarray(irx_pct, dtype=float) / 100.0
    r = -np.log1p(-d * 91.0 / 360.0) * 365.0 / 91.0
    return float(r) if r.ndim == 0 else r


def latest_snapshot_dir() -> Path:
    dirs = sorted(p for p in RAW_DIR.glob("*") if (p / "meta.json").exists())
    if not dirs:
        raise FileNotFoundError("No cached snapshot found; run `python -m volproj.run_all --refresh`.")
    return dirs[-1]


class Snapshot:
    """Read-only view of a cached snapshot folder."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else latest_snapshot_dir()
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.asof = date.fromisoformat(self.meta["asof"])

    @property
    def rate(self) -> float:
        """Flat continuously-compounded risk-free rate from the 13-week T-bill (see irx_to_cc)."""
        return irx_to_cc(self.meta["irx_pct"])

    def spot(self, sym: str) -> float:
        return self.meta["tickers"][sym]["spot"]

    def div_yield(self, sym: str) -> float:
        """Continuous dividend yield q = ln(1 + trailing-12m dividends / spot)."""
        t = self.meta["tickers"][sym]
        return float(np.log1p(t["dividends_trailing_12m"] / t["spot"]))

    def market_cap(self, sym: str) -> float | None:
        return self.meta["tickers"][sym]["market_cap"]

    def chain(self, sym: str) -> pd.DataFrame:
        df = pd.read_csv(self.path / "chains" / f"{sym}.csv")
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        return df

    def history(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "history_spy_vix_irx.csv", index_col="date", parse_dates=True)

    def earnings_dates(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "earnings_dates.csv")
        df["earnings_ts"] = pd.to_datetime(df["earnings_ts"], utc=True).dt.tz_convert("America/New_York")
        return df

    def surface_history(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "history_surface_adjclose.csv", index_col="date", parse_dates=True)

    def basket_history(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "history_basket_adjclose.csv", index_col="date", parse_dates=True)
