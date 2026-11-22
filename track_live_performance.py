"""
track_live_performance.py — Live Performance Tracker for The Value Hunter (Alpaca Paper).

Tracks the live performance of the Alpaca Paper trading account against
SPY (S&P 500) and QQQ (Nasdaq 100) benchmarks starting from October 21, 2026.

Key Functions:
  1. Connects to Alpaca API to fetch current total portfolio value, cash, and positions.
  2. Fetches real-time benchmark prices for SPY and QQQ via yfinance fast_info['last_price'].
  3. Logs daily portfolio values to `live_portfolio_history.csv`.
  4. Plots a normalized performance comparison graph (%) comparing The Value Hunter (Paper) vs. SPY vs. QQQ
     and saves it as `live_vs_benchmarks.png`.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Try importing environment loader and credentials from alpaca_execution or config
try:
    from alpaca_execution import get_alpaca_credentials, init_trading_client
except ImportError:
    def get_alpaca_credentials():
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT, ALPACA_PAPER
        return ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT, ALPACA_PAPER

    def init_trading_client(dry_run: bool = False):
        api_key, secret_key, endpoint, paper = get_alpaca_credentials()
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=endpoint)

from config import INITIAL_CAPITAL

# Setup Logger
logger = logging.getLogger("track_live")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

INCEPTION_DATE = "2026-10-21"
SPY_INCEPTION_PRICE = 773.09
QQQ_INCEPTION_PRICE = 725.20
DEFAULT_HISTORY_CSV = "live_portfolio_history.csv"
DEFAULT_CHART_PNG = "live_vs_benchmarks.png"


# ── Alpaca Live Portfolio Fetcher ──────────────────────────────────────────────

def fetch_alpaca_portfolio() -> Dict[str, any]:
    """
    Connect to Alpaca Paper Trading API and fetch current portfolio status.
    Returns:
        dict containing equity, cash, buying_power, positions_value,
        account_status, trade_date, and timestamp_str.
    """
    client = init_trading_client(dry_run=False)
    account = client.get_account()

    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    last_equity = float(account.last_equity) if getattr(account, "last_equity", None) else equity

    # Fetch positions to calculate aggregate market value
    positions = client.get_all_positions()
    positions_value = sum(float(p.market_value) for p in positions)

    # Resolve market date using Alpaca clock
    try:
        clock = client.get_clock()
        market_dt = clock.timestamp
        trade_date = market_dt.strftime("%Y-%m-%d")
        timestamp_str = market_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        is_open = clock.is_open
    except Exception as e:
        logger.warning("Could not fetch Alpaca clock (%s), using local/UTC time.", e)
        now = datetime.now()
        trade_date = now.strftime("%Y-%m-%d")
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        is_open = False

    return {
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
        "last_equity": last_equity,
        "positions_value": positions_value,
        "positions_count": len(positions),
        "trade_date": trade_date,
        "timestamp_str": timestamp_str,
        "account_number": account.account_number,
        "status": str(account.status),
        "is_open": is_open,
    }


# ── Benchmark Fetcher ──────────────────────────────────────────────────────────

def fetch_benchmarks(start_date: str = INCEPTION_DATE) -> Dict[str, float]:
    """
    Fetch current/latest real-time benchmark prices for SPY and QQQ directly via
    yfinance fast_info['last_price'].
    Returns:
        Dict[str, float] with keys 'SPY' and 'QQQ' containing live float prices.
    """
    logger.info("Fetching live benchmark prices for SPY and QQQ via yfinance fast_info...")

    spy_price = float(yf.Ticker("SPY").fast_info['last_price'])
    qqq_price = float(yf.Ticker("QQQ").fast_info['last_price'])

    logger.info("Live benchmark quotes -> SPY: $%.2f | QQQ: $%.2f", spy_price, qqq_price)

    return {
        "SPY": spy_price,
        "QQQ": qqq_price,
    }


# ── CSV Logging & History Management ──────────────────────────────────────────

def update_portfolio_history(
    alpaca_info: Dict[str, any],
    benchmark_prices: Dict[str, float],
    csv_path: str = DEFAULT_HISTORY_CSV,
    initial_capital: float = INITIAL_CAPITAL,
    start_date: str = INCEPTION_DATE,
) -> pd.DataFrame:
    """
    Record or update the daily portfolio equity and live benchmark levels in `live_portfolio_history.csv`.
    Maintains clean daily records and calculates cumulative return percentages.
    """
    trade_date = alpaca_info["trade_date"]
    equity = alpaca_info["equity"]
    cash = alpaca_info["cash"]
    positions_value = alpaca_info["positions_value"]
    timestamp_str = alpaca_info["timestamp_str"]

    # Retrieve live SPY and QQQ float prices
    spy_price = float(benchmark_prices["SPY"])
    qqq_price = float(benchmark_prices["QQQ"])

    # Load existing CSV if available
    if os.path.exists(csv_path):
        try:
            df_hist = pd.read_csv(csv_path)
            # Ensure proper string formatting for date
            df_hist["date"] = df_hist["date"].astype(str)
            # Exclude non-trading days (weekends: Saturday=5, Sunday=6)
            if not df_hist.empty and "date" in df_hist.columns:
                df_hist = df_hist[pd.to_datetime(df_hist["date"]).dt.dayofweek < 5].copy()
        except Exception as e:
            logger.warning("Could not read existing history CSV (%s). Creating new.", e)
            df_hist = pd.DataFrame()
    else:
        df_hist = pd.DataFrame()

    # Determine baseline prices on inception date from history CSV (or fallback constants)
    if not df_hist.empty and "date" in df_hist.columns:
        if start_date in df_hist["date"].values:
            base_row = df_hist[df_hist["date"] == start_date].iloc[0]
            spy_base = float(base_row["spy_price"])
            qqq_base = float(base_row["qqq_price"])
        else:
            spy_base = float(df_hist["spy_price"].iloc[0])
            qqq_base = float(df_hist["qqq_price"].iloc[0])
    elif trade_date == start_date:
        spy_base = spy_price
        qqq_base = qqq_price
    else:
        spy_base = SPY_INCEPTION_PRICE
        qqq_base = QQQ_INCEPTION_PRICE

    # Skip appending weekend records if script is executed on a weekend
    if pd.to_datetime(trade_date).dayofweek >= 5:
        logger.warning("Trade date %s falls on a weekend; skipping appending to history CSV.", trade_date)
        return df_hist

    # Calculate returns relative to inception
    fund_return_pct = round(((equity / initial_capital) - 1.0) * 100.0, 4)
    spy_return_pct = round(((spy_price / spy_base) - 1.0) * 100.0, 4)
    qqq_return_pct = round(((qqq_price / qqq_base) - 1.0) * 100.0, 4)

    new_row = {
        "date": trade_date,
        "portfolio_value": round(equity, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "spy_price": round(spy_price, 4),
        "qqq_price": round(qqq_price, 4),
        "fund_return_pct": fund_return_pct,
        "spy_return_pct": spy_return_pct,
        "qqq_return_pct": qqq_return_pct,
        "updated_at": timestamp_str,
    }

    if not df_hist.empty and "date" in df_hist.columns:
        if trade_date in df_hist["date"].values:
            # Update existing date entry
            idx = df_hist.index[df_hist["date"] == trade_date][0]
            for col, val in new_row.items():
                df_hist.at[idx, col] = val
            logger.info("Updated existing history entry for %s.", trade_date)
        else:
            # Append new trading day
            df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
            logger.info("Logged new history entry for %s.", trade_date)
    else:
        df_hist = pd.DataFrame([new_row])
        logger.info("Created history log with initial entry for %s.", trade_date)

    # Sort chronologically by date
    df_hist.sort_values(by="date", inplace=True)
    df_hist.reset_index(drop=True, inplace=True)

    # Save to CSV
    df_hist.to_csv(csv_path, index=False)
    logger.info("Portfolio history saved -> %s", csv_path)

    return df_hist


# ── Performance Visualization ─────────────────────────────────────────────────

def plot_live_performance(
    df_hist: pd.DataFrame,
    benchmark_prices: Dict[str, float],
    alpaca_info: Dict[str, any],
    output_png: str = DEFAULT_CHART_PNG,
    initial_capital: float = INITIAL_CAPITAL,
    start_date: str = INCEPTION_DATE,
) -> None:
    """
    Generate an institutional dark-theme normalized performance comparison chart
    comparing The Value Hunter (Paper) vs. SPY vs. QQQ.
    All assets are normalized to a $100,000 starting capital on Inception Date.
    """
    logger.info("Generating live performance chart (normalized to $100k starting capital)...")

    # Color Palette
    COLOR_BG = "#0e1118"
    COLOR_PANEL = "#161b26"
    COLOR_GRID = "#252d3d"
    COLOR_TEXT = "#e2e8f0"
    COLOR_TEXT_MUTED = "#94a3b8"

    COLOR_FUND = "#00e5ff"    # Vibrant Cyan
    COLOR_SPY = "#ff7043"     # Sunset Coral
    COLOR_QQQ = "#b388ff"     # Lavender Purple

    # Exclude weekend entries and sort chronologically
    df_clean = df_hist.copy()
    if not df_clean.empty and "date" in df_clean.columns:
        df_clean = df_clean[pd.to_datetime(df_clean["date"]).dt.dayofweek < 5].copy()
    df_clean.sort_values(by="date", inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    dates = []
    fund_returns = []
    spy_returns = []
    qqq_returns = []

    for i, (_, row) in enumerate(df_clean.iterrows()):
        d_str = str(row["date"])
        f_ret = float(row["fund_return_pct"])
        s_ret = float(row["spy_return_pct"])
        q_ret = float(row["qqq_return_pct"])

        dt = pd.to_datetime(d_str)
        day_name = dt.strftime("%A")  # Wednesday, Thursday, Friday, Monday

        # Format label with day of week and date
        if i == len(df_clean) - 1 and len(df_clean) > 1:
            lbl = f"{day_name}\n{d_str}\n(Live)"
        elif i == 0 or d_str == start_date:
            lbl = f"{day_name}\n{d_str}\n(Inception)"
        else:
            lbl = f"{day_name}\n{d_str}"

        dates.append(lbl)
        fund_returns.append(f_ret)
        spy_returns.append(s_ret)
        qqq_returns.append(q_ret)

    if not dates:
        dates = [f"Inception\n{start_date}"]
        fund_returns = [0.0]
        spy_returns = [0.0]
        qqq_returns = [0.0]

    # Current metrics and normalized benchmark dollar values ($100k base)
    cur_equity = alpaca_info["equity"]
    cur_fund_ret = fund_returns[-1]
    cur_spy_ret = spy_returns[-1]
    cur_qqq_ret = qqq_returns[-1]

    cur_spy_val = initial_capital * (1.0 + cur_spy_ret / 100.0)
    cur_qqq_val = initial_capital * (1.0 + cur_qqq_ret / 100.0)

    # Initialize Figure
    fig = plt.figure(figsize=(15, 9), facecolor=COLOR_BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.4, 1.2], width_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # ── Panel 1: Normalized Performance Curve ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLOR_PANEL)

    x_indices = np.arange(len(dates))

    # Plot Curves with Dollar Value in labels
    ax1.plot(
        x_indices, fund_returns,
        color=COLOR_FUND, lw=2.8, marker="o", markersize=7,
        label=f"The Value Hunter: ${cur_equity:,.2f} ({cur_fund_ret:+.2f}%)", zorder=4
    )
    ax1.plot(
        x_indices, spy_returns,
        color=COLOR_SPY, lw=2.0, marker="s", markersize=6, linestyle="--",
        label=f"SPY S&P 500: ${cur_spy_val:,.2f} ({cur_spy_ret:+.2f}%)", zorder=3
    )
    ax1.plot(
        x_indices, qqq_returns,
        color=COLOR_QQQ, lw=2.0, marker="^", markersize=6, linestyle=":",
        label=f"QQQ Nasdaq 100: ${cur_qqq_val:,.2f} ({cur_qqq_ret:+.2f}%)", zorder=3
    )

    # Fill Fund vs Zero
    ax1.fill_between(
        x_indices, fund_returns, 0,
        color=COLOR_FUND, alpha=0.12, zorder=2
    )

    # Baseline 0.0% ($100k) line
    ax1.axhline(0, color=COLOR_TEXT_MUTED, linestyle="--", lw=1.0, alpha=0.6, zorder=1)

    # Titles & Labels
    ax1.set_title(
        "THE VALUE HUNTER vs BENCHMARKS — LIVE PAPER PERFORMANCE ($100k BASE)",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, pad=12, parse_math=False
    )
    ax1.set_ylabel("Normalized Return (%)", fontsize=11, color=COLOR_TEXT)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(dates, fontsize=9.5, color=COLOR_TEXT)
    ax1.tick_params(colors=COLOR_TEXT, which="both")
    ax1.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.7)

    # Format Left Y-axis with + / - signs
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.2f}%"))

    # Ensure comfortable headroom so curves and annotations never collide with top title
    y_min, y_max = ax1.get_ylim()
    y_span = max(abs(y_max - y_min), 0.3)
    ax1.set_ylim(y_min - y_span * 0.10, y_max + y_span * 0.28)

    # Secondary Y-axis on Right: Portfolio Dollar Value ($100k Base)
    ax1_right = ax1.twinx()
    ax1_right.set_facecolor("none")
    y_lim_min, y_lim_max = ax1.get_ylim()
    ax1_right.set_ylim(initial_capital * (1.0 + y_lim_min / 100.0), initial_capital * (1.0 + y_lim_max / 100.0))

    # Align right axis ticks exactly with left axis ticks
    left_ticks = ax1.get_yticks()
    ax1_right.set_yticks([initial_capital * (1.0 + t / 100.0) for t in left_ticks])
    ax1_right.set_yticklabels([f"${initial_capital * (1.0 + t / 100.0):,.0f}" for t in left_ticks], fontsize=9.5, color=COLOR_TEXT)
    ax1_right.set_ylabel("Portfolio Dollar Value ($100k Base)", fontsize=11, color=COLOR_TEXT, labelpad=12, parse_math=False)
    ax1_right.tick_params(colors=COLOR_TEXT, which="both")
    ax1_right.grid(False)
    for spine in ax1_right.spines.values():
        spine.set_color(COLOR_GRID)

    # Legend
    legend = ax1.legend(
        loc="upper left",
        facecolor=COLOR_PANEL,
        edgecolor=COLOR_GRID,
        fontsize=9.5,
        labelcolor=COLOR_TEXT,
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.2)

    # Current value callout box (safely positioned inside the chart area)
    last_x = x_indices[-1]
    y_cur_min, y_cur_max = ax1.get_ylim()
    y_mid = (y_cur_min + y_cur_max) / 2.0
    offset_y = -40 if cur_fund_ret >= y_mid else 28
    offset_x = -35 if len(dates) > 1 else 0

    callout_text = (
        f"The Value Hunter: ${cur_equity:,.2f} ({cur_fund_ret:+.2f}%)\n"
        f"SPY: ${cur_spy_val:,.2f} ({cur_spy_ret:+.2f}%)  |  QQQ: ${cur_qqq_val:,.2f} ({cur_qqq_ret:+.2f}%)"
    )

    ax1.annotate(
        callout_text,
        xy=(last_x, cur_fund_ret),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_FUND,
            lw=1.4,
            shrinkA=2,
            shrinkB=5,
        ),
        bbox=dict(boxstyle="round,pad=0.55", fc=COLOR_BG, ec=COLOR_FUND, lw=1.5, alpha=0.95),
        color=COLOR_TEXT,
        fontsize=9.5,
        fontweight="bold",
        ha="right" if len(dates) > 1 else "center",
        va="top" if offset_y < 0 else "bottom",
        parse_math=False,
        zorder=6,
    )

    for spine in ax1.spines.values():
        spine.set_color(COLOR_GRID)

    # ── Panel 2: Total Return Comparison Bar Chart ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLOR_PANEL)

    bar_labels = ["The Value Hunter\n(Paper)", "SPY\n(S&P 500)", "QQQ\n(Nasdaq 100)"]
    bar_values = [cur_fund_ret, cur_spy_ret, cur_qqq_ret]
    bar_dollars = [cur_equity, cur_spy_val, cur_qqq_val]
    bar_colors = [COLOR_FUND, COLOR_SPY, COLOR_QQQ]

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, width=0.55, alpha=0.9, zorder=3)
    ax2.axhline(0, color=COLOR_TEXT_MUTED, lw=0.8, alpha=0.6, zorder=2)

    # Data value labels on bars: return % + dollar value
    for bar, d_val in zip(bars, bar_dollars):
        h = bar.get_height()
        va_mode = "bottom" if h >= 0 else "top"
        offset = 0.02 if h >= 0 else -0.03
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            f"{h:+.2f}%\n(${d_val:,.0f})",
            ha="center",
            va=va_mode,
            color=COLOR_TEXT,
            fontsize=9.5,
            fontweight="bold",
            parse_math=False,
        )

    ax2.set_title("Current Cumulative Return & Value ($100k Base)", fontsize=11, fontweight="semibold", color=COLOR_TEXT, pad=8, parse_math=False)
    ax2.set_ylabel("Return (%)", fontsize=10, color=COLOR_TEXT)
    ax2.tick_params(colors=COLOR_TEXT, which="both")
    ax2.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.2f}%"))
    b_min, b_max = min(bar_values), max(bar_values)
    b_span = max(abs(b_max - b_min), 0.5)
    ax2.set_ylim(min(b_min - b_span * 0.35, -0.3), max(b_max + b_span * 0.35, 0.3))

    for spine in ax2.spines.values():
        spine.set_color(COLOR_GRID)

    # ── Panel 3: Key Portfolio & Alpha Metrics Card ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLOR_PANEL)
    ax3.axis("off")

    alpha_spy = cur_fund_ret - cur_spy_ret
    alpha_qqq = cur_fund_ret - cur_qqq_ret
    alpha_spy_dollar = cur_equity - cur_spy_val
    alpha_qqq_dollar = cur_equity - cur_qqq_val

    alpha_spy_dollar_str = f"+${alpha_spy_dollar:,.2f}" if alpha_spy_dollar >= 0 else f"-${abs(alpha_spy_dollar):,.2f}"
    alpha_qqq_dollar_str = f"+${alpha_qqq_dollar:,.2f}" if alpha_qqq_dollar >= 0 else f"-${abs(alpha_qqq_dollar):,.2f}"

    metrics_text = [
        ("Inception Date", f"{start_date}"),
        ("Initial Capital ($100k Base)", f"${initial_capital:,.2f}"),
        ("The Value Hunter Value", f"${alpaca_info['equity']:,.2f} ({cur_fund_ret:+.2f}%)"),
        ("SPY Portfolio Value", f"${cur_spy_val:,.2f} ({cur_spy_ret:+.2f}%)"),
        ("QQQ Portfolio Value", f"${cur_qqq_val:,.2f} ({cur_qqq_ret:+.2f}%)"),
        ("Alpha vs SPY", f"{alpha_spy:+.2f}% ({alpha_spy_dollar_str})"),
        ("Alpha vs QQQ", f"{alpha_qqq:+.2f}% ({alpha_qqq_dollar_str})"),
        ("Cash Balance", f"${alpaca_info['cash']:,.2f}"),
        ("Invested Holdings", f"${alpaca_info['positions_value']:,.2f} ({alpaca_info['positions_count']} stocks)"),
        ("Alpaca Buying Power", f"${alpaca_info['buying_power']:,.2f}"),
    ]

    ax3.set_title("Live Portfolio Metrics & Alpha ($100k Base)", fontsize=11, fontweight="semibold", color=COLOR_TEXT, pad=8, parse_math=False)

    y_pos = 0.88
    for label, val in metrics_text:
        is_highlight = "Alpha" in label or "The Value Hunter Value" in label
        val_color = COLOR_FUND if is_highlight else COLOR_TEXT
        font_weight = "bold" if is_highlight else "normal"

        ax3.text(0.04, y_pos, label, transform=ax3.transAxes, color=COLOR_TEXT_MUTED, fontsize=9.2, parse_math=False)
        ax3.text(0.96, y_pos, val, transform=ax3.transAxes, color=val_color, fontsize=9.2, fontweight=font_weight, ha="right", parse_math=False)
        y_pos -= 0.092

    # Border frame for ax3
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes, fill=False, edgecolor=COLOR_GRID, lw=1.2)
    ax3.add_patch(rect)

    # ── Brand Watermark (VH monogram) ─────────────────────────────────────────
    fig.text(
        0.5, 0.66, "VH", ha="center", va="center",
        fontsize=150, fontweight="bold", color=COLOR_FUND, alpha=0.05,
    )
    fig.text(
        0.995, 0.01, "THE VALUE HUNTER  |  VH", ha="right", va="bottom",
        fontsize=9, fontweight="semibold", color=COLOR_TEXT_MUTED, alpha=0.85,
    )

    # Save Figure
    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    logger.info("Live performance graph saved -> %s", output_png)


# ── Console Report ────────────────────────────────────────────────────────────

def print_live_report(
    alpaca_info: Dict[str, any],
    df_hist: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    start_date: str = INCEPTION_DATE,
) -> None:
    """Print an executive status report to terminal."""
    last_row = df_hist.iloc[-1]
    fund_ret = float(last_row["fund_return_pct"])
    spy_ret = float(last_row["spy_return_pct"])
    qqq_ret = float(last_row["qqq_return_pct"])

    cur_equity = alpaca_info["equity"]
    cur_spy_val = initial_capital * (1.0 + spy_ret / 100.0)
    cur_qqq_val = initial_capital * (1.0 + qqq_ret / 100.0)

    alpha_spy = fund_ret - spy_ret
    alpha_qqq = fund_ret - qqq_ret
    alpha_spy_dollar = cur_equity - cur_spy_val
    alpha_qqq_dollar = cur_equity - cur_qqq_val

    alpha_spy_str = f"{alpha_spy:+.2f}% (+${alpha_spy_dollar:,.2f})" if alpha_spy_dollar >= 0 else f"{alpha_spy:+.2f}% (-${abs(alpha_spy_dollar):,.2f})"
    alpha_qqq_str = f"{alpha_qqq:+.2f}% (+${alpha_qqq_dollar:,.2f})" if alpha_qqq_dollar >= 0 else f"{alpha_qqq:+.2f}% (-${abs(alpha_qqq_dollar):,.2f})"

    print("\n" + "=" * 82)
    print("  THE VALUE HUNTER — LIVE PAPER PERFORMANCE vs BENCHMARKS ($100k BASE)")
    print("=" * 82)
    print(f"  Inception Date      : {start_date}")
    print(f"  Current Date (ET)   : {alpaca_info['trade_date']}  (Market Open: {alpaca_info['is_open']})")
    print(f"  Alpaca Status       : Account {alpaca_info['account_number']} ({alpaca_info['status']})")
    print(f"  Last Updated        : {alpaca_info['timestamp_str']}")
    print("-" * 82)
    print(f"  Initial Capital     : ${initial_capital:>12,.2f}  (Normalized baseline across all assets)")
    print(f"  Current Total Equity: ${alpaca_info['equity']:>12,.2f}")
    print(f"  Invested Holdings   : ${alpaca_info['positions_value']:>12,.2f}  ({alpaca_info['positions_count']} positions)")
    print(f"  Cash Balance        : ${alpaca_info['cash']:>12,.2f}")
    print(f"  Buying Power        : ${alpaca_info['buying_power']:>12,.2f}")
    print("-" * 82)
    print(f"  {'Asset':<22} {'Portfolio Value ($100k)':>24} {'Return (%)':>14} {'Alpha vs Strategy':>20}")
    print("-" * 82)
    print(f"  {'The Value Hunter':<22} {f'${cur_equity:,.2f}':>24} {f'{fund_ret:+.2f}%':>14} {'--':>20}")
    print(f"  {'SPY (S&P 500)':<22} {f'${cur_spy_val:,.2f}':>24} {f'{spy_ret:+.2f}%':>14} {alpha_spy_str:>20}")
    print(f"  {'QQQ (Nasdaq 100)':<22} {f'${cur_qqq_val:,.2f}':>24} {f'{qqq_ret:+.2f}%':>14} {alpha_qqq_str:>20}")
    print("=" * 82)
    print(f"  Historical Log      : {DEFAULT_HISTORY_CSV} ({len(df_hist)} records)")
    print(f"  Performance Graph   : {DEFAULT_CHART_PNG}")
    print("=" * 82 + "\n")


# ── Execution Pipeline & Continuous Monitoring ────────────────────────────────

def run_pipeline(args) -> Tuple[Dict[str, any], pd.DataFrame]:
    """Execute a single run of the performance tracking pipeline."""
    alpaca_info = fetch_alpaca_portfolio()
    benchmark_prices = fetch_benchmarks(start_date=args.start_date)
    df_hist = update_portfolio_history(
        alpaca_info=alpaca_info,
        benchmark_prices=benchmark_prices,
        csv_path=args.history_file,
        initial_capital=args.initial_capital,
        start_date=args.start_date,
    )
    if not args.no_chart:
        plot_live_performance(
            df_hist=df_hist,
            benchmark_prices=benchmark_prices,
            alpaca_info=alpaca_info,
            output_png=args.chart_file,
            initial_capital=args.initial_capital,
            start_date=args.start_date,
        )
    return alpaca_info, df_hist


def run_watch_loop(args):
    """Run an automated continuous monitoring loop every `interval` seconds."""
    print("\n" + "=" * 78, flush=True)
    print("  THE VALUE HUNTER — LIVE CONTINUOUS MONITORING MODE", flush=True)
    print(f"  Polling Interval : {args.interval}s", flush=True)
    print(f"  History File     : {args.history_file}", flush=True)
    print(f"  Chart Output     : {args.chart_file}", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    print("=" * 78 + "\n", flush=True)

    try:
        while True:
            # Check market status via Alpaca clock or Eastern time
            try:
                client = init_trading_client(dry_run=False)
                clock = client.get_clock()
                is_open = clock.is_open
                market_dt = clock.timestamp
                time_str = market_dt.strftime("%H:%M:%S")
            except Exception as e:
                is_open = False
                time_str = datetime.now().strftime("%H:%M:%S")

            if is_open:
                try:
                    alpaca_info, df_hist = run_pipeline(args)
                    last_row = df_hist.iloc[-1]
                    f_ret = float(last_row["fund_return_pct"])
                    s_ret = float(last_row["spy_return_pct"])
                    q_ret = float(last_row["qqq_return_pct"])
                    cur_eq = alpaca_info["equity"]
                    s_val = args.initial_capital * (1.0 + s_ret / 100.0)
                    q_val = args.initial_capital * (1.0 + q_ret / 100.0)

                    print(
                        f"[{time_str} ET] The Value Hunter: ${cur_eq:,.2f} ({f_ret:+.2f}%) | "
                        f"SPY: ${s_val:,.2f} ({s_ret:+.2f}%) | "
                        f"QQQ: ${q_val:,.2f} ({q_ret:+.2f}%) | Status: Market OPEN",
                        flush=True
                    )
                except Exception as ex:
                    logger.error("Error during live watch iteration: %s", ex)
            else:
                print(f"[{time_str} ET] Market is currently CLOSED. Waiting for next session...", flush=True)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Live monitoring stopped by user (Ctrl+C). Exiting cleanly.\n", flush=True)
        sys.exit(0)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track live performance of Alpaca Paper trading account against SPY and QQQ."
    )
    parser.add_argument(
        "--history-file",
        default=DEFAULT_HISTORY_CSV,
        help=f"Path to history CSV file (default: {DEFAULT_HISTORY_CSV})",
    )
    parser.add_argument(
        "--chart-file",
        default=DEFAULT_CHART_PNG,
        help=f"Path to output performance graph PNG (default: {DEFAULT_CHART_PNG})",
    )
    parser.add_argument(
        "--start-date",
        default=INCEPTION_DATE,
        help=f"Inception date for tracking (default: {INCEPTION_DATE})",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=INITIAL_CAPITAL,
        help=f"Initial capital baseline in USD (default: {INITIAL_CAPITAL})",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation and only update history CSV",
    )
    parser.add_argument(
        "--watch", "--live",
        dest="watch",
        action="store_true",
        help="Run in live continuous monitoring mode (auto-refresh loop)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Refresh interval in seconds for --watch mode (default: 60)",
    )

    args = parser.parse_args()

    if args.watch:
        run_watch_loop(args)
    else:
        alpaca_info, df_hist = run_pipeline(args)
        print_live_report(
            alpaca_info=alpaca_info,
            df_hist=df_hist,
            initial_capital=args.initial_capital,
            start_date=args.start_date,
        )


if __name__ == "__main__":
    main()
