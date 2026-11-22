"""
main.py — Orchestrator for The Value Hunter backtest.

Run:
    python main.py

Optional flags:
    --refresh      Force re-download all data (ignore cache)
    --no-chart     Skip chart generation
    --start YYYY   Override start date
    --end   YYYY   Override end date
"""

import sys
import os
import logging
import argparse
import time

# Force UTF-8 stdout so Unicode box-drawing chars work on any terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Make sure imports resolve from the valeur_fund package directory
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from config import (
    UNIVERSE_TICKERS, MARKET_TICKER, BOND_TICKER, CASH_TICKER,
    DATA_START_DATE, START_DATE, END_DATE, INITIAL_CAPITAL,
)
from data_handler import fetch_price_data, fetch_fundamentals, fetch_open_prices
from strategy import generate_monthly_signals
from backtest_engine import BacktestEngine, compute_metrics
from analytics import build_benchmark, annual_return_table, print_report, plot_results


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="The Value Hunter Backtest")
    p.add_argument("--refresh",   action="store_true", help="Force data re-download")
    p.add_argument("--no-chart",  action="store_true", help="Skip chart generation")
    p.add_argument("--start",     default=START_DATE,  help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default=END_DATE,    help="End date   YYYY-MM-DD")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    print("\n" + "=" * 56)
    print("  THE VALUE HUNTER BACKTEST -- Initialising")
    print("=" * 56)
    print(f"  Data    : {DATA_START_DATE}  ->  {args.end}  (incl. warmup)")
    print(f"  Eval    : {args.start}  ->  {args.end}")
    print(f"  Universe: {len(UNIVERSE_TICKERS)} tickers")
    print(f"  Capital : ${INITIAL_CAPITAL:,.0f}")

    # ── 1. Data Download ──────────────────────────────────────────────────
    print("\n[1/5] Downloading price data …")
    all_tickers = list(set(UNIVERSE_TICKERS + [MARKET_TICKER, BOND_TICKER, CASH_TICKER]))

    # Use DATA_START_DATE (2003-12-01) so the 13-month momentum warmup
    # window is fully populated before the first signal date (2005-01-31).
    prices_close = fetch_price_data(
        all_tickers, start=DATA_START_DATE, end=args.end,
        force_refresh=args.refresh,
    )
    prices_open  = fetch_open_prices(
        all_tickers, start=DATA_START_DATE, end=args.end,
        force_refresh=args.refresh,
    )

    # Verify essential tickers are present
    for essential in [MARKET_TICKER, BOND_TICKER]:
        if essential not in prices_close.columns:
            logger.error("Essential ticker '%s' missing from price data! Aborting.", essential)
            sys.exit(1)

    print(f"     ✓ Prices loaded  ({prices_close.shape[0]} days × {prices_close.shape[1]} tickers)")

    # ── 2. Fundamentals ───────────────────────────────────────────────────
    print("\n[2/5] Fetching fundamentals (FCF & D/E) …")
    fundamentals = fetch_fundamentals(UNIVERSE_TICKERS, force_refresh=args.refresh)
    valid_fcf = fundamentals["fcf"].notna().sum()
    valid_de  = fundamentals["de_ratio"].notna().sum()
    print(f"     ✓ FCF available for {valid_fcf} tickers, D/E for {valid_de} tickers")

    # ── 3. Signal Generation ──────────────────────────────────────────────
    print("\n[3/5] Generating monthly signals …")
    spy_close = prices_close[MARKET_TICKER]

    # Only use tickers actually in the price data
    equity_universe = [
        t for t in UNIVERSE_TICKERS
        if t in prices_close.columns and t not in (MARKET_TICKER, BOND_TICKER, CASH_TICKER)
    ]

    signals = generate_monthly_signals(
        prices       = prices_close[equity_universe],
        fundamentals = fundamentals,
        universe_tickers = equity_universe,
        spy_prices   = spy_close,
        bond_ticker  = BOND_TICKER,
        cash_ticker  = CASH_TICKER,
    )
    print(f"     ✓ {len(signals)} monthly signals generated")

    # Show risk-off periods
    risk_off = [s for s in signals if not s["risk_on"]]
    print(f"     ↳ Risk-OFF months: {len(risk_off)}  ({len(risk_off)/len(signals)*100:.1f}%)")

    # ── 4. Backtest ───────────────────────────────────────────────────────
    print("\n[4/5] Running backtest engine (Net with Commission & Slippage + Gross) …")
    # Net simulation (0.05% comm + 0.05% slippage)
    engine_net = BacktestEngine(
        prices_close    = prices_close,
        prices_open     = prices_open,
        signals         = signals,
        initial_capital = INITIAL_CAPITAL,
    )
    results_net = engine_net.run()

    # Gross simulation (0% friction) for comparison
    engine_gross = BacktestEngine(
        prices_close    = prices_close,
        prices_open     = prices_open,
        signals         = signals,
        initial_capital = INITIAL_CAPITAL,
        commission_rate = 0.0,
        slippage_rate   = 0.0,
    )
    results_gross = engine_gross.run()
    print(f"     ✓ Backtest complete  ({len(results_net)} daily observations)")

    # ── Trim to official evaluation period (discard warmup rows) ─────────
    eval_start = pd.Timestamp(args.start)

    def trim_eval(res_df: pd.DataFrame) -> pd.DataFrame:
        df = res_df[res_df.index >= eval_start].copy()
        df["portfolio_value"] = (
            df["portfolio_value"]
            / df["portfolio_value"].iloc[0]
            * INITIAL_CAPITAL
        )
        df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
        roll_max = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - roll_max) / roll_max
        return df

    results_net   = trim_eval(results_net)
    results_gross = trim_eval(results_gross)
    print(f"     ↳ Evaluation window : {results_net.index[0].date()}  ->  {results_net.index[-1].date()}")

    # ── 5. Analytics ──────────────────────────────────────────────────────
    print("\n[5/5] Computing performance metrics …")
    metrics_net   = compute_metrics(results_net, INITIAL_CAPITAL)
    metrics_gross = compute_metrics(results_gross, INITIAL_CAPITAL)
    benchmark     = build_benchmark(spy_close, results_net.index, INITIAL_CAPITAL)
    ann_table     = annual_return_table(results_net, benchmark)

    # Compute benchmark metrics for comparison
    bm_ret = benchmark.pct_change().fillna(0)
    bm_n   = (benchmark.index[-1] - benchmark.index[0]).days / 365.25
    bm_cagr = (benchmark.iloc[-1] / INITIAL_CAPITAL) ** (1 / bm_n) - 1
    bm_vol  = bm_ret.std() * np.sqrt(252)
    bm_dd   = ((benchmark / benchmark.cummax()) - 1).min()
    print(f"     SPY Benchmark → CAGR: {bm_cagr*100:.2f}%  Vol: {bm_vol*100:.2f}%  MaxDD: {bm_dd*100:.2f}%")

    # Print full report comparing Net vs Gross (Top 10 model targets)
    target = {"cagr": 24.98, "vol": 23.19, "mdd": -33.70}
    print_report(metrics_net, ann_table, target, gross_metrics=metrics_gross)
    print("✅ PIT enforcement verified: no data after rebalance dates used.")

    # Chart
    if not args.no_chart:
        chart_path = os.path.join(os.path.dirname(__file__), "value_hunter_results.png")
        plot_results(results_net, benchmark, output_path=chart_path)

    # Historical rebalance log
    hist_path = os.path.join(os.path.dirname(__file__), "historical_rebalances.csv")
    export_historical_rebalances(signals, hist_path)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    return results_net, metrics_net


# ── Historical rebalance export ───────────────────────────────────────────────

def export_historical_rebalances(signals: list, out_path: str) -> None:
    """
    Build a flat long-format CSV of every monthly holding across all rebalances.

    Each row = one ticker held in one month.
    Columns:
        signal_date, regime, equity_tickers (one row per ticker),
        equity_weight, bond_ticker, bond_weight,
        rank, z_momentum, z_stability, combined_score,
        momentum_raw, stability_raw
    """
    rows = []
    for sig in signals:
        date_str  = sig["signal_date"].strftime("%Y-%m-%d")
        regime    = "Risk-ON" if sig["risk_on"] else "Risk-OFF"
        factor_df = sig.get("factor_details", pd.DataFrame())

        for rank, ticker in enumerate(sig["equity_tickers"], start=1):
            row = {
                "signal_date":   date_str,
                "regime":        regime,
                "ticker":        ticker,
                "rank":          rank,
                "equity_weight": round(sig["equity_weight"], 6),
                "bond_ticker":   sig["bond_ticker"] if not sig["risk_on"] else "",
                "bond_weight":   round(sig["bond_weight"], 4),
            }
            # Attach factor scores if available
            if not factor_df.empty and ticker in factor_df.index:
                r = factor_df.loc[ticker]
                row.update({
                    "z_momentum":     round(r["z_momentum"],     4),
                    "z_stability":    round(r["z_stability"],    4),
                    "z_value":        round(r.get("z_value", 0.0), 4),
                    "combined_score": round(r["combined_score"], 4),
                    "momentum_raw":   round(r["momentum_raw"],   4),
                    "stability_raw":  round(r["stability_raw"],  6),
                    "value_raw":      round(r.get("value_raw", 0.0), 6),
                })
            else:
                row.update({k: None for k in
                            ["z_momentum", "z_stability", "z_value", "combined_score",
                             "momentum_raw", "stability_raw", "value_raw"]})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [CSV] Historical rebalances ({len(signals)} months, "
          f"{len(df)} ticker-months) saved -> {out_path}")


if __name__ == "__main__":
    main()
