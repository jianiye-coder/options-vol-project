"""Static matplotlib figures for the README (PNG, light background)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (validated adjacent + all-pairs)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal ramp, light->dark
NEG, POS = "#e34948", "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 10,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def smiles(surfaces: dict, summaries: dict, path, target_days=(14, 30, 60, 120, 240)):
    fig, axes = plt.subplots(1, len(surfaces), figsize=(5 * len(surfaces), 4.2), sharey=False)
    for ax, (sym, surf) in zip(axes, surfaces.items()):
        es = summaries[sym]
        chosen = []
        for d in target_days:
            e = es.iloc[(es["days"] - d).abs().argsort().iloc[0]]["expiry"]
            if e not in chosen:
                chosen.append(e)
        ramp = BLUES[-len(chosen):] if len(chosen) <= len(BLUES) else BLUES
        for color, exp in zip(ramp, chosen):
            g = surf[(surf["expiry"] == exp) & (surf["k"].abs() <= 0.3)].sort_values("k")
            days = int(round(g["T"].iloc[0] * 365))
            ax.plot(g["k"], g["iv"], color=color, label=f"{days}d", marker="o", markersize=2.5, linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title(f"{sym} implied vol smile")
        ax.set_xlabel("log-moneyness  ln(K / F)")
        _pct(ax)
        ax.legend(title="expiry", fontsize=8, title_fontsize=8, loc="upper right")
    axes[0].set_ylabel("implied vol (OTM mids)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def term_structure(summaries: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for color, (sym, es) in zip(SERIES, summaries.items()):
        es = es.dropna(subset=["atm_iv"])
        a1.plot(es["days"], es["atm_iv"], color=color, marker="o", markersize=4, label=sym)
        a2.plot(es["days"], es["put_skew25"], color=color, marker="o", markersize=4, label=sym)
        a1.annotate(sym, (es["days"].iloc[-1], es["atm_iv"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    a1.set_title("ATM implied vol term structure")
    a1.set_xlabel("days to expiry")
    a1.set_ylabel("ATM implied vol")
    _pct(a1)
    a2.set_title("Put skew: 25-delta put IV minus ATM IV")
    a2.set_xlabel("days to expiry")
    a2.set_ylabel("vol points")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.1f}"))
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks([7, 14, 30, 60, 120, 240, 365])
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def greeks_profile(surf: pd.DataFrame, sym: str, expiry, path):
    g = surf[(surf["expiry"] == expiry) & (surf["k"].abs() <= 0.15)].sort_values("strike")
    S = g["S"].iloc[0]
    days = int(round(g["T"].iloc[0] * 365))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    specs = [("delta", "delta  (per $1)"), ("gamma", "gamma  (per $1)"),
             ("vega", "vega  ($ per vol pt)"), ("theta", "theta  ($ per day)")]
    for ax, (col, title) in zip(axes, specs):
        for color, kind in ((SERIES[1], "put"), (SERIES[0], "call")):
            h = g[g["type"] == kind]
            ax.plot(h["strike"], h[col], color=color, label=f"OTM {kind}s")
        ax.axvline(S, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("strike")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{sym} Greeks across strikes, {expiry} expiry ({days}d), each at its own implied vol; dotted = spot",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hedge_examples(examples: dict, path):
    fig, axes = plt.subplots(1, len(examples), figsize=(5.5 * len(examples), 4), sharey=False)
    for ax, (title, sim) in zip(np.atleast_1d(axes), examples.items()):
        x = np.arange(1, len(sim) + 1)
        prem = sim["V"].iloc[0]
        ax.plot(x, sim["gamma_pnl"].cumsum() / prem, color=SERIES[0], label="gamma  ½ΓΔS²")
        ax.plot(x, (sim["theta_pnl"] + sim["carry_pnl"]).cumsum() / prem, color=SERIES[1], label="theta + carry")
        ax.plot(x, sim["total"].cumsum() / prem, color=INK, linewidth=2.4, label="total hedged P&L")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("trading day")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        _pct(ax)
    np.atleast_1d(axes)[0].set_ylabel("cumulative P&L, % of option premium")
    np.atleast_1d(axes)[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hedge_scatter(bt: pd.DataFrame, path):
    x = (bt["realized"] - bt["implied"]) * 100
    y = bt["pnl_vol_pts"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(x, y, s=22, color=np.where(y > 0, POS, NEG), edgecolor=SURFACE, linewidth=0.8, zorder=3)
    b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK2, linewidth=1.2, linestyle="--", label=f"OLS slope {b[0]:.2f}")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("realized vol − implied vol (VIX) over the month, vol points")
    ax.set_ylabel("hedged P&L ÷ entry vega  (vol points)")
    ax.set_title(f"Delta-hedged 1M ATM SPY call, {len(bt)} monthly windows")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def index_vs_single(tbl: pd.DataFrame, index_row: dict, path):
    t = tbl.sort_values("put_skew_norm")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(t["ticker"], t["put_skew_norm"], color=SERIES[0], height=0.7, label="single stock")
    ax.barh(["SPY (index)"], [index_row["put_skew_norm"]], color=SERIES[1], height=0.7, label="SPY")
    wavg = np.average(t["put_skew_norm"], weights=t["weight"])
    ax.axvline(wavg, color=INK2, linestyle="--", linewidth=1.2)
    ax.annotate(f"cap-weighted basket avg {wavg:.2f}", (wavg, 0.5), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
    ax.set_xlabel("(25Δ put IV − ATM IV) ÷ ATM IV")
    ax.set_title("Index put skew vs. its largest members (same expiry)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def earnings_moves(events: dict, path):
    fig, axes = plt.subplots(1, len(events), figsize=(5.5 * len(events), 3.8))
    for ax, (sym, ev) in zip(np.atleast_1d(axes), events.items()):
        hist = ev["realized"]
        x = np.arange(len(hist))
        moves = hist["move"].to_numpy()
        ax.bar(x, np.abs(moves), color=np.where(moves > 0, POS, NEG), width=0.7)
        if np.isfinite(ev["implied"].get("implied_abs_move", np.nan)):
            ax.axhline(ev["implied"]["implied_abs_move"], color=SERIES[1], linewidth=2,
                       label=f"implied E|move| {ev['implied']['implied_abs_move'] * 100:.1f}%")
        ax.axhline(np.abs(moves).mean(), color=INK2, linewidth=1.2, linestyle="--",
                   label=f"historical mean |move| {np.abs(moves).mean() * 100:.1f}%")
        ax.set_xticks(x, [pd.Timestamp(d).strftime("%y-%m") for d in hist["reaction_day"]], rotation=60, fontsize=7.5)
        ax.set_title(f"{sym}: earnings-day moves (blue up, red down)")
        ax.set_ylabel("|1-day log return|")
        _pct(ax)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
