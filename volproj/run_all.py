"""End-to-end pipeline: cached snapshot -> surfaces, Greeks, backtest, correlation -> output/ and figures/.

    python -m volproj.run_all            # use the latest cached snapshot (offline)
    python -m volproj.run_all --refresh  # download a fresh snapshot first (Yahoo, no key needed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import corr, events, hedge, plots
from .data import BASKET_TICKERS, ROOT, SURFACE_TICKERS, Snapshot, snapshot
from .surface import (build_surface, expiry_summary, interp_by_days, model_free_vol,
                      parity_forwards, prepare_chain, same_strike_iv_gap)

OUT = ROOT / "output"
FIG = ROOT / "figures"


def _common_expiry(snap: Snapshot, syms, target_days: int):
    sets = [set(snap.chain(s)["expiry"].unique()) for s in syms]
    common = sorted(set.intersection(*sets))
    return min(common, key=lambda e: abs((e - snap.asof).days - target_days))


def vix_replication(snap: Snapshot, spy_summary: pd.DataFrame) -> dict:
    raw = snap.chain("SPY")
    pts = []
    for e in sorted(raw["expiry"].unique()):
        d = (e - snap.asof).days
        if 15 <= d <= 50:
            pts.append((d, model_free_vol(raw, snap.asof, snap.spot("SPY"), snap.rate, e)))
    d, v = np.array(pts).T
    mf30 = float(np.sqrt(np.interp(30, d, v**2 * d) / 30))
    return {"model_free_30d": mf30, "published_vix": snap.meta["vix_pct"] / 100,
            "spy_atm_30d": interp_by_days(spy_summary, "atm_iv", 30),
            "per_expiry": {int(di): float(vi) for di, vi in pts}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="download a new snapshot before running")
    ap.add_argument("--snapshot", type=Path, help="path to a specific data/raw/<date> folder")
    args = ap.parse_args(argv)

    if args.refresh:
        print("Downloading snapshot ...")
        snapshot()
    snap = Snapshot(args.snapshot)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    print(f"Snapshot {snap.asof}  r={snap.rate:.4f}  VIX={snap.meta['vix_pct']:.2f}")
    summary: dict = {"asof": str(snap.asof), "rate_cc": snap.rate}

    # 1-3. Surfaces, IV, Greeks
    surfaces, summaries = {}, {}
    for sym in SURFACE_TICKERS:
        raw = snap.chain(sym)
        q = snap.div_yield(sym)
        surf = build_surface(raw, snap.asof, snap.spot(sym), snap.rate, q)
        es = expiry_summary(surf)
        pf = parity_forwards(prepare_chain(raw, snap.asof, snap.spot(sym), snap.rate, q))
        surf.to_csv(OUT / f"surface_{sym}.csv", index=False)
        es.to_csv(OUT / f"expiry_summary_{sym}.csv", index=False)
        pf.to_csv(OUT / f"parity_forwards_{sym}.csv", index=False)
        surfaces[sym], summaries[sym] = surf, es
        summary[sym] = {
            "spot": snap.spot(sym), "div_yield_trailing": q, "n_contracts_raw": int(len(raw)),
            "n_contracts_used": int(len(surf)), "iv_methods": surf["iv_method"].value_counts().to_dict(),
            "max_abs_reprice_error": float(surf["model_minus_mid"].abs().max()),
            "median_abs_iv_vs_yahoo_pts": float((surf["iv"] - surf["yahoo_iv"]).abs().median() * 100),
            "median_abs_parity_gap_bp": float(pf["gap_bp"].abs().median()),
            "atm_30d": interp_by_days(es, "atm_iv", 30), "atm_90d": interp_by_days(es, "atm_iv", 90),
            "atm_180d": interp_by_days(es, "atm_iv", 180),
            "put_skew25_30d": float(np.interp(30, es["days"], es["put_skew25"])),
            "rr25_30d": float(np.interp(30, es["days"], es["rr25"])),
            # Same-strike call-minus-put IV near the forward (median |gap|, vol pts). With the
            # parity forward this is ~0 by construction; with the dividend-yield forward it
            # measures how wrong that forward is.
            "atm_call_put_iv_gap_pts": {
                mode: same_strike_iv_gap(raw, snap.asof, snap.spot(sym), snap.rate, q, mode)
                for mode in ("div", "parity")
            },
        }
        print(f"  {sym}: {len(surf)} OTM quotes, ATM30={summary[sym]['atm_30d']:.3f}")

    summary["vix_replication"] = vix_replication(snap, summaries["SPY"])
    plots.smiles(surfaces, summaries, FIG / "smiles.png")
    plots.term_structure(summaries, FIG / "term_structure.png")
    spy_es = summaries["SPY"]
    g_exp = spy_es.iloc[(spy_es["days"] - 30).abs().argsort().iloc[0]]["expiry"]
    plots.greeks_profile(surfaces["SPY"], "SPY", g_exp, FIG / "greeks_spy.png")

    # Earnings: implied vs realized moves
    ed = snap.earnings_dates()
    px = snap.surface_history()
    ev_out = {}
    for sym in ed["ticker"].unique():
        ts = ed.loc[ed["ticker"] == sym, "earnings_ts"]
        nxt = ts[ts.dt.date > snap.asof].min()
        implied = (events.implied_event_move(summaries[sym], nxt.date(), after_close=nxt.hour >= 12)
                   if pd.notna(nxt) else {})
        realized = events.realized_earnings_moves(px[sym], ts, snap.asof).sort_values("reaction_day").tail(12)
        ev_out[sym] = {"implied": implied, "realized": realized}
        realized.to_csv(OUT / f"earnings_moves_{sym}.csv", index=False)
        summary[f"earnings_{sym}"] = {
            "next_earnings": str(nxt.date()) if pd.notna(nxt) else None,
            **{k: (str(v) if not isinstance(v, float) else v) for k, v in implied.items()},
            "hist_mean_abs_move": float(realized["move"].abs().mean()),
            "hist_median_abs_move": float(realized["move"].abs().median()),
            "hist_n": int(len(realized)),
            "hist_share_above_implied": float((realized["move"].abs() > implied.get("implied_abs_move", np.nan)).mean()),
        }
    plots.earnings_moves(ev_out, FIG / "earnings_moves.png")

    # 4. Delta-hedged P&L backtest
    hist = snap.history()
    bt = hedge.rolling_backtest(hist)
    bt.to_csv(OUT / "hedge_backtest.csv", index=False)
    summary["hedge_backtest"] = hedge.summarize_backtest(bt)
    summary["hedge_sensitivity"] = {
        f"{s:+.2f}": hedge.summarize_backtest(hedge.rolling_backtest(hist, vol_shift=s))
        for s in (-0.02, -0.03, -0.04)
    }
    h = hist.dropna(subset=["spy_close", "vix", "irx"])

    def window(entry):
        return hedge.window_from_history(h, h.index.get_loc(pd.Timestamp(entry)))[0]

    calm = bt.iloc[(bt["pnl_vol_pts"] - bt["pnl_vol_pts"].median()).abs().argsort().iloc[0]]
    wild = bt.loc[bt["pnl_vol_pts"].idxmax()]
    plots.hedge_examples({
        f"Typical month (median P&L ÷ vega): entry {calm['entry']}\nimplied {calm['implied']:.0%}, realized {calm['realized']:.0%}": window(calm["entry"]),
        f"Best month by P&L ÷ vega: entry {wild['entry']}\nimplied {wild['implied']:.0%}, realized {wild['realized']:.0%}": window(wild["entry"]),
    }, FIG / "hedge_examples.png")
    plots.hedge_scatter(bt, FIG / "hedge_scatter.png")

    # 5. Index vs single-stock skew and implied correlation
    corr_rows = []
    for days in (30, 60):
        exp = _common_expiry(snap, ["SPY"] + BASKET_TICKERS, days)
        per = []
        for sym in ["SPY"] + BASKET_TICKERS:
            raw = snap.chain(sym)
            raw = raw[raw["expiry"] == exp]
            surf = build_surface(raw, snap.asof, snap.spot(sym), snap.rate, snap.div_yield(sym))
            row = expiry_summary(surf)
            if row.empty:
                continue
            row = row.iloc[0]
            per.append({"ticker": sym, "expiry": exp, "atm_iv": row["atm_iv"], "put25_iv": row["put25_iv"],
                        "call25_iv": row["call25_iv"], "put25_k": row["put25_k"],
                        "market_cap": snap.market_cap(sym)})
        per = pd.DataFrame(per)
        idx_row = per[per["ticker"] == "SPY"].iloc[0]
        names = per[(per["ticker"] != "SPY")].dropna(subset=["atm_iv", "put25_iv", "market_cap"]).copy()
        names["weight"] = names["market_cap"] / names["market_cap"].sum()
        names["put_skew"] = names["put25_iv"] - names["atm_iv"]
        names["put_skew_norm"] = names["put_skew"] / names["atm_iv"]
        rho_atm = corr.implied_correlation(idx_row["atm_iv"], names["atm_iv"], names["weight"])
        rho_put = corr.implied_correlation(idx_row["put25_iv"], names["put25_iv"], names["weight"])
        names.to_csv(OUT / f"basket_{days}d.csv", index=False)
        corr_rows.append({
            "target_days": days, "expiry": str(exp), "days": (exp - snap.asof).days,
            "n_names": int(len(names)), "basket_weight_names": names["ticker"].tolist(),
            "spy_atm": idx_row["atm_iv"], "spy_put25": idx_row["put25_iv"],
            "spy_put_skew": idx_row["put25_iv"] - idx_row["atm_iv"],
            "spy_put_skew_norm": (idx_row["put25_iv"] - idx_row["atm_iv"]) / idx_row["atm_iv"],
            "basket_wavg_atm": float(np.average(names["atm_iv"], weights=names["weight"])),
            "basket_wavg_put_skew": float(np.average(names["put_skew"], weights=names["weight"])),
            "basket_wavg_put_skew_norm": float(np.average(names["put_skew_norm"], weights=names["weight"])),
            "share_names_flatter_than_spy_norm": float((names["put_skew_norm"] < (idx_row["put25_iv"] - idx_row["atm_iv"]) / idx_row["atm_iv"]).mean()),
            "implied_corr_atm": rho_atm, "implied_corr_put25": rho_put,
            # 25-delta puts sit at different moneyness for low- and high-vol names.
            "spy_put25_k": idx_row["put25_k"],
            "basket_wavg_put25_k": float(np.average(names["put25_k"], weights=names["weight"])),
        })
        if days == 30:
            plots.index_vs_single(names, {"put_skew_norm": corr_rows[-1]["spy_put_skew_norm"]}, FIG / "index_vs_single_skew.png")

    bh = snap.basket_history()
    rets = np.log(bh).diff().dropna().tail(63)
    w = pd.Series({s: snap.market_cap(s) for s in BASKET_TICKERS}, dtype=float)
    realized = corr.realized_correlation(rets[BASKET_TICKERS], rets["SPY"], w)
    summary["implied_correlation"] = corr_rows
    summary["realized_correlation_63d"] = realized
    pd.DataFrame(corr_rows).drop(columns="basket_weight_names").to_csv(OUT / "implied_correlation.csv", index=False)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("vix_replication", "hedge_backtest", "implied_correlation",
                                               "realized_correlation_63d")}, indent=2, default=str))
    for sym in ("AAPL", "NVDA"):
        print(sym, json.dumps(summary.get(f"earnings_{sym}"), default=str))


if __name__ == "__main__":
    main()
