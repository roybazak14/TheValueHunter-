"""
analytics.py — Reporting, chart generation, and benchmark comparison.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from config import INITIAL_CAPITAL, TRADING_DAYS_YEAR

logger = logging.getLogger(__name__)


# ── Benchmark helper ──────────────────────────────────────────────────────────

def build_benchmark(
    spy_close: pd.Series,
    results_index: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """Return a buy-and-hold SPY equity curve aligned to results_index."""
    spy = spy_close.reindex(results_index, method="ffill")
    spy = spy / spy.iloc[0] * initial_capital
    return spy


# ── Annual return table ───────────────────────────────────────────────────────

def annual_return_table(
    results: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build a year-by-year return comparison table."""
    ret   = results["daily_return"]
    annual_fund = ((1 + ret).resample("YE").prod() - 1) * 100

    rows = {"Fund (%)": annual_fund.round(2)}
    if benchmark is not None:
        bm_ret    = benchmark.pct_change().fillna(0)
        annual_bm = ((1 + bm_ret).resample("YE").prod() - 1) * 100
        rows["SPY (%)"] = annual_bm.round(2)

    df = pd.DataFrame(rows)
    df.index = df.index.year
    df.index.name = "Year"
    return df


# ── Console report ────────────────────────────────────────────────────────────

def print_report(
    metrics: dict,
    ann_table: pd.DataFrame,
    target: dict,
    gross_metrics: Optional[dict] = None,
) -> None:
    """Print a formatted performance report to the terminal, optionally comparing Net vs Gross."""
    width = 68 if gross_metrics else 56
    sep   = "=" * width

    print(f"\n{'=' * width}")
    print(f"  THE VALUE HUNTER -- BACKTEST RESULTS {'(NET vs GROSS)' if gross_metrics else ''}")
    print(f"{'=' * width}")
    print(f"  Period : {metrics['Start']}  ->  {metrics['End']}")
    print(f"  Years  : {metrics['Years']}")
    print(sep)

    metrics_order = [
        "CAGR (%)",
        "Total Return (%)",
        "Annual Vol (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Calmar Ratio",
        "Best Year (%)",
        "Worst Year (%)",
    ]

    targets = {
        "CAGR (%)":         ("~24.98", target.get("cagr", 24.98)),
        "Annual Vol (%)":   ("~23.19", target.get("vol",  23.19)),
        "Max Drawdown (%)": ("~-33.70", -abs(target.get("mdd", 33.70))),
    }

    if gross_metrics:
        print(f"  {'Metric':<20} {'Net (Costs)':>12} {'Gross':>12} {'Target':>10}  {'Status':>6}")
        print("-" * width)
        for key in metrics_order:
            val_net = metrics[key]
            val_gross = gross_metrics.get(key, "N/A")
            if key in targets:
                label_target, t_val = targets[key]
                diff = val_net - t_val
                flag = "[OK]" if abs(diff) < 5 else "[!!]"
                print(f"  {key:<20} {str(val_net):>12} {str(val_gross):>12} {label_target:>10}  {flag:>6}")
            else:
                print(f"  {key:<20} {str(val_net):>12} {str(val_gross):>12}")
    else:
        for key in metrics_order:
            val = metrics[key]
            if key in targets:
                label_target, t_val = targets[key]
                diff = val - t_val
                flag = "[OK]" if abs(diff) < 5 else "[!!]"
                print(f"  {key:<22} {str(val):>10}    target {label_target:>8}  {flag}")
            else:
                print(f"  {key:<22} {str(val):>10}")

    print(sep)
    print("\n  Annual Returns (Net)\n")
    print(ann_table.to_string())
    print(f"\n{'=' * width}\n")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_results(
    results: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    output_path: str = "value_hunter_results.png",
) -> None:
    """Save a 3-panel chart: equity curve, drawdown, annual returns."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.5, 1.5], hspace=0.35)

    color_fund = "#00e5ff"
    color_bm   = "#ff6b35"
    color_bg   = "#0f0f1a"
    color_grid = "#2a2a3a"

    # Compute summary stats for legend
    pv = results["portfolio_value"]
    n_yrs = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr_pct = ((pv.iloc[-1] / pv.iloc[0]) ** (1.0 / n_yrs) - 1.0) * 100.0
    final_val = pv.iloc[-1]
    fund_dd_min = results["drawdown"].min() * 100.0

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(color_bg)
    ax1.semilogy(
        results.index, results["portfolio_value"], color=color_fund,
        lw=2.2, label=f"The Value Hunter (CAGR: {cagr_pct:.2f}% | Final: ${final_val:,.0f})",
    )
    if benchmark is not None:
        bm_cagr = ((benchmark.iloc[-1] / benchmark.iloc[0]) ** (1.0 / n_yrs) - 1.0) * 100.0
        ax1.semilogy(
            benchmark.index, benchmark.values, color=color_bm,
            lw=1.6, alpha=0.85, label=f"SPY Buy & Hold (CAGR: {bm_cagr:.2f}% | Final: ${benchmark.iloc[-1]:,.0f})",
        )
    ax1.set_title(
        "THE VALUE HUNTER — Historical Equity Curve (2005–2025, 30% Mom / 30% Stab / 40% Value)",
        color="white", fontsize=13.5, fontweight="bold", pad=12, parse_math=False,
    )
    ax1.set_ylabel("Portfolio Value ($, Log Scale)", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, color=color_grid, linestyle="--", alpha=0.6)
    ax1.legend(facecolor=color_bg, edgecolor=color_grid, labelcolor="white", fontsize=10.5)
    ax1.spines[:].set_color(color_grid)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Shade 2008 & 2022 crisis periods
    for start_yr, end_yr in [("2008-01", "2009-03"), ("2022-01", "2022-12")]:
        ax1.axvspan(pd.Timestamp(start_yr), pd.Timestamp(end_yr),
                    alpha=0.15, color="red", label="_nolegend_")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(color_bg)
    ax2.fill_between(
        results.index, results["drawdown"] * 100, 0,
        color=color_fund, alpha=0.4, label=f"The Value Hunter DD (Max: {fund_dd_min:.2f}%)",
    )
    if benchmark is not None:
        bm_dd = (benchmark / benchmark.cummax() - 1) * 100
        ax2.fill_between(
            benchmark.index, bm_dd, 0,
            color=color_bm, alpha=0.25, label=f"SPY DD (Max: {bm_dd.min():.2f}%)",
        )
    ax2.set_title("Underwater Drawdown Profile (%)", color="white", fontsize=11.5, fontweight="bold", pad=6)
    ax2.set_ylabel("Drawdown (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, color=color_grid, linestyle="--", alpha=0.6)
    ax2.legend(facecolor=color_bg, edgecolor=color_grid, labelcolor="white", fontsize=10)
    ax2.spines[:].set_color(color_grid)

    # ── Panel 3: Annual Returns Bar Chart ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(color_bg)
    annual = ((1 + results["daily_return"]).resample("YE").prod() - 1) * 100
    years  = annual.index.year
    colors = [color_fund if v >= 0 else "#ff4444" for v in annual.values]
    bars = ax3.bar(years, annual.values, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, annual.values):
        va_m = "bottom" if val >= 0 else "top"
        off = 1.2 if val >= 0 else -1.8
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0, val + off,
            f"{val:+.1f}%", ha="center", va=va_m, color="white", fontsize=8,
        )
    ax3.axhline(0, color="white", lw=0.8, alpha=0.5)
    ax3.set_title("Annual Calendar Returns (%)", color="white", fontsize=11.5, fontweight="bold", pad=6)
    ax3.set_ylabel("Return (%)", color="white")
    ax3.set_xticks(years)
    ax3.tick_params(colors="white")
    ax3.set_ylim(min(annual.min() - 12, -20), max(annual.max() + 12, 85))
    ax3.grid(True, color=color_grid, linestyle="--", alpha=0.4, axis="y")
    ax3.spines[:].set_color(color_grid)

    # Brand watermark & footer
    fig.text(
        0.5, 0.72, "VH", ha="center", va="center",
        fontsize=140, fontweight="bold", color=color_fund, alpha=0.04,
    )
    fig.text(
        0.99, 0.005, "THE VALUE HUNTER  |  VH", ha="right", va="bottom",
        fontsize=9.5, fontweight="bold", color="#94a3b8", alpha=0.85,
    )

    # Save primary output path and legacy aliases (.png and .jpg)
    base_dir = os.path.dirname(os.path.abspath(output_path))
    target_files = [
        output_path,
        os.path.join(base_dir, "valeur_fund_results.png"),
        os.path.join(base_dir, "valeur_fund_results.jpg"),
    ]
    seen_paths = set()
    for p in target_files:
        norm_p = os.path.normcase(os.path.abspath(p))
        if norm_p not in seen_paths:
            seen_paths.add(norm_p)
            plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=color_bg)
            logger.info("Chart saved → %s", p)
            print(f"  [Chart] Saved to: {p}")
    plt.close(fig)
