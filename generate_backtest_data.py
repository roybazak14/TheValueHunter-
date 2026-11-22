"""
generate_backtest_data.py — Computes and exports high-performance JSON backtest payload.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    UNIVERSE_TICKERS, MARKET_TICKER, BOND_TICKER, CASH_TICKER,
    DATA_START_DATE, START_DATE, END_DATE, INITIAL_CAPITAL,
)
from data_handler import fetch_price_data, fetch_fundamentals, fetch_open_prices
from strategy import generate_monthly_signals
from backtest_engine import BacktestEngine, compute_metrics
from analytics import build_benchmark, annual_return_table


def generate_backtest_payload(output_path: str = "backtest_results.json"):
    print("[1/4] Loading cached price and fundamental data ...")
    all_tickers = list(set(UNIVERSE_TICKERS + [MARKET_TICKER, BOND_TICKER, CASH_TICKER]))
    prices_close = fetch_price_data(all_tickers, start=DATA_START_DATE, end=END_DATE, force_refresh=False)
    prices_open  = fetch_open_prices(all_tickers, start=DATA_START_DATE, end=END_DATE, force_refresh=False)
    fundamentals = fetch_fundamentals(UNIVERSE_TICKERS, force_refresh=False)

    print("[2/4] Generating strategy signals & running net / gross backtests ...")
    spy_close = prices_close[MARKET_TICKER]
    equity_universe = [
        t for t in UNIVERSE_TICKERS
        if t in prices_close.columns and t not in (MARKET_TICKER, BOND_TICKER, CASH_TICKER)
    ]

    signals = generate_monthly_signals(
        prices=prices_close[equity_universe],
        fundamentals=fundamentals,
        universe_tickers=equity_universe,
        spy_prices=spy_close,
        bond_ticker=BOND_TICKER,
        cash_ticker=CASH_TICKER,
    )

    # Net engine (Audited Modern Alpaca: 0% commissions, 0.10% slippage, 1% cash buffer)
    engine_net = BacktestEngine(
        prices_close=prices_close,
        prices_open=prices_open,
        signals=signals,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.0,
        slippage_rate=0.0010,
        cash_buffer=0.01,
    )
    res_net = engine_net.run()

    # Gross engine
    engine_gross = BacktestEngine(
        prices_close=prices_close,
        prices_open=prices_open,
        signals=signals,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    res_gross = engine_gross.run()

    # Trim to evaluation window (2005-01-01 onwards)
    eval_start = pd.Timestamp(START_DATE)
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

    res_net = trim_eval(res_net)
    res_gross = trim_eval(res_gross)
    benchmark = build_benchmark(spy_close, res_net.index, INITIAL_CAPITAL)
    bm_roll_max = benchmark.cummax()
    bm_drawdown = (benchmark - bm_roll_max) / bm_roll_max

    # Load and build QQQ benchmark
    qqq_pkl = "cache/qqq_historical.pkl"
    if os.path.exists(qqq_pkl):
        with open(qqq_pkl, "rb") as f:
            qqq_close = pickle.load(f)
    else:
        import yfinance as yf
        df_qqq = yf.download("QQQ", start="2003-12-01", end=END_DATE, progress=False)
        if isinstance(df_qqq.columns, pd.MultiIndex):
            qqq_close = df_qqq["Close"]["QQQ"].dropna()
        else:
            qqq_close = df_qqq["Close"].dropna()
        with open(qqq_pkl, "wb") as f:
            pickle.dump(qqq_close, f)

    benchmark_qqq = build_benchmark(qqq_close, res_net.index, INITIAL_CAPITAL)
    bm_qqq_roll_max = benchmark_qqq.cummax()
    bm_qqq_drawdown = (benchmark_qqq - bm_qqq_roll_max) / bm_qqq_roll_max

    print("[3/4] Computing performance metrics and annual distributions ...")
    metrics_net = compute_metrics(res_net, INITIAL_CAPITAL)
    metrics_gross = compute_metrics(res_gross, INITIAL_CAPITAL)
    ann_table = annual_return_table(res_net, benchmark)

    # SPY metrics
    bm_ret = benchmark.pct_change().fillna(0)
    bm_n = (benchmark.index[-1] - benchmark.index[0]).days / 365.25
    bm_cagr = (benchmark.iloc[-1] / INITIAL_CAPITAL) ** (1 / bm_n) - 1
    bm_vol = bm_ret.std() * np.sqrt(252)
    bm_dd = float(bm_drawdown.min())
    bm_sharpe = float((bm_cagr - 0.02) / bm_vol) if bm_vol > 0 else 0.0
    bm_tot_ret = float((benchmark.iloc[-1] / INITIAL_CAPITAL - 1) * 100)

    # QQQ metrics
    qqq_ret = benchmark_qqq.pct_change().fillna(0)
    qqq_n = (benchmark_qqq.index[-1] - benchmark_qqq.index[0]).days / 365.25
    qqq_cagr = (benchmark_qqq.iloc[-1] / INITIAL_CAPITAL) ** (1 / qqq_n) - 1
    qqq_vol = qqq_ret.std() * np.sqrt(252)
    qqq_dd = float(bm_qqq_drawdown.min())
    qqq_sharpe = float((qqq_cagr - 0.02) / qqq_vol) if qqq_vol > 0 else 0.0
    qqq_tot_ret = float((benchmark_qqq.iloc[-1] / INITIAL_CAPITAL - 1) * 100)
    annual_qqq_series = ((1 + qqq_ret).resample("YE").prod() - 1) * 100

    # Format annual returns list
    annual_returns = []
    for yr, row in ann_table.iterrows():
        f_ret = float(row["Fund (%)"])
        s_ret = float(row["SPY (%)"])
        q_ret = 0.0
        matching_q = [val for idx, val in annual_qqq_series.items() if idx.year == int(yr)]
        if matching_q:
            q_ret = float(matching_q[0])

        annual_returns.append({
            "year": int(yr),
            "fund_return_pct": round(f_ret, 2),
            "spy_return_pct": round(s_ret, 2),
            "qqq_return_pct": round(q_ret, 2),
            "alpha_pct": round(f_ret - s_ret, 2),
            "alpha_qqq_pct": round(f_ret - q_ret, 2),
        })

    # Prepare daily series
    daily_dates = [d.strftime("%Y-%m-%d") for d in res_net.index]
    daily_fund_val = [round(float(v), 2) for v in res_net["portfolio_value"]]
    daily_spy_val = [round(float(v), 2) for v in benchmark]
    daily_qqq_val = [round(float(v), 2) for v in benchmark_qqq]
    daily_fund_dd = [round(float(v) * 100.0, 2) for v in res_net["drawdown"]]
    daily_spy_dd = [round(float(v) * 100.0, 2) for v in bm_drawdown]
    daily_qqq_dd = [round(float(v) * 100.0, 2) for v in bm_qqq_drawdown]

    # Prepare monthly resampled series (lighter weight for fast default rendering)
    m_fund = res_net["portfolio_value"].resample("ME").last()
    m_spy = benchmark.resample("ME").last()
    m_qqq = benchmark_qqq.resample("ME").last()
    m_fund_dd = res_net["drawdown"].resample("ME").last() * 100.0
    m_spy_dd = bm_drawdown.resample("ME").last() * 100.0
    m_qqq_dd = bm_qqq_drawdown.resample("ME").last() * 100.0

    monthly_dates = [d.strftime("%Y-%m-%d") for d in m_fund.index]
    monthly_fund_val = [round(float(v), 2) for v in m_fund]
    monthly_spy_val = [round(float(v), 2) for v in m_spy]
    monthly_qqq_val = [round(float(v), 2) for v in m_qqq]
    monthly_fund_dd = [round(float(v), 2) for v in m_fund_dd]
    monthly_spy_dd = [round(float(v), 2) for v in m_spy_dd]
    monthly_qqq_dd = [round(float(v), 2) for v in m_qqq_dd]

    # Prepare yearly resampled series (annual year-end snapshots across 21 calendar years)
    y_fund = res_net["portfolio_value"].resample("YE").last()
    y_spy = benchmark.resample("YE").last()
    y_qqq = benchmark_qqq.resample("YE").last()
    y_fund_dd = res_net["drawdown"].resample("YE").last() * 100.0
    y_spy_dd = bm_drawdown.resample("YE").last() * 100.0
    y_qqq_dd = bm_qqq_drawdown.resample("YE").last() * 100.0

    yearly_dates = [d.strftime("%Y-%m-%d") for d in y_fund.index]
    yearly_fund_val = [round(float(v), 2) for v in y_fund]
    yearly_spy_val = [round(float(v), 2) for v in y_spy]
    yearly_qqq_val = [round(float(v), 2) for v in y_qqq]
    yearly_fund_dd = [round(float(v), 2) for v in y_fund_dd]
    yearly_spy_dd = [round(float(v), 2) for v in y_spy_dd]
    yearly_qqq_dd = [round(float(v), 2) for v in y_qqq_dd]

    payload = {
        "status": "ok",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": {
            "start": res_net.index[0].strftime("%Y-%m-%d"),
            "end": res_net.index[-1].strftime("%Y-%m-%d"),
            "years": round(float(metrics_net["Years"]), 2),
            "trading_days": len(res_net),
        },
        "strategy_specs": {
            "universe": "Nasdaq-100 & S&P Large-Cap (101 constituents)",
            "selection_size": 10,
            "weights": "Equal-Weight (10% each)",
            "rebalance": "Monthly at Open",
            "factors": {
                "value": "40% (TTM FCF Yield, capped at 20%)",
                "stability": "30% (90-day Price Volatility Inverse)",
                "momentum": "30% (12M - 1M Momentum, minimum > 0.0%)",
            },
            "macro_overlay": "SPY > 10-Month SMA (Risk-ON: 100% Equities; Risk-OFF: 50% TLT Bonds + 50% Equities)",
            "friction": "Audited Modern Alpaca: 0.10% slippage, 1% cash buffer, zero commissions (Net)",
        },
        "summary_metrics": {
            "strategy_net": {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": round(float(res_net["portfolio_value"].iloc[-1]), 2),
                "total_return_pct": round(float(metrics_net["Total Return (%)"]), 2),
                "cagr_pct": round(float(metrics_net["CAGR (%)"]), 2),
                "annual_vol_pct": round(float(metrics_net["Annual Vol (%)"]), 2),
                "sharpe_ratio": round(float(metrics_net["Sharpe Ratio"]), 3),
                "max_drawdown_pct": round(float(metrics_net["Max Drawdown (%)"]), 2),
                "calmar_ratio": round(float(metrics_net["Calmar Ratio"]), 3),
                "best_year_pct": round(float(metrics_net["Best Year (%)"]), 2),
                "worst_year_pct": round(float(metrics_net["Worst Year (%)"]), 2),
            },
            "strategy_gross": {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": round(float(res_gross["portfolio_value"].iloc[-1]), 2),
                "total_return_pct": round(float(metrics_gross["Total Return (%)"]), 2),
                "cagr_pct": round(float(metrics_gross["CAGR (%)"]), 2),
                "annual_vol_pct": round(float(metrics_gross["Annual Vol (%)"]), 2),
                "sharpe_ratio": round(float(metrics_gross["Sharpe Ratio"]), 3),
                "max_drawdown_pct": round(float(metrics_gross["Max Drawdown (%)"]), 2),
                "calmar_ratio": round(float(metrics_gross["Calmar Ratio"]), 3),
            },
            "benchmark_spy": {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": round(float(benchmark.iloc[-1]), 2),
                "total_return_pct": round(bm_tot_ret, 2),
                "cagr_pct": round(bm_cagr * 100.0, 2),
                "annual_vol_pct": round(bm_vol * 100.0, 2),
                "sharpe_ratio": round(bm_sharpe, 3),
                "max_drawdown_pct": round(bm_dd * 100.0, 2),
                "calmar_ratio": round(abs((bm_cagr * 100.0) / (bm_dd * 100.0)), 3) if bm_dd != 0 else 0.0,
            },
            "benchmark_qqq": {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": round(float(benchmark_qqq.iloc[-1]), 2),
                "total_return_pct": round(qqq_tot_ret, 2),
                "cagr_pct": round(qqq_cagr * 100.0, 2),
                "annual_vol_pct": round(qqq_vol * 100.0, 2),
                "sharpe_ratio": round(qqq_sharpe, 3),
                "max_drawdown_pct": round(qqq_dd * 100.0, 2),
                "calmar_ratio": round(abs((qqq_cagr * 100.0) / (qqq_dd * 100.0)), 3) if qqq_dd != 0 else 0.0,
            },
            "alpha": {
                "cagr_spread_pct": round(float(metrics_net["CAGR (%)"]) - (bm_cagr * 100.0), 2),
                "cagr_spread_qqq_pct": round(float(metrics_net["CAGR (%)"]) - (qqq_cagr * 100.0), 2),
                "max_drawdown_reduction_pct": round(abs(bm_dd * 100.0) - abs(float(metrics_net["Max Drawdown (%)"])), 2),
                "max_drawdown_reduction_qqq_pct": round(abs(qqq_dd * 100.0) - abs(float(metrics_net["Max Drawdown (%)"])), 2),
                "sharpe_improvement": round(float(metrics_net["Sharpe Ratio"]) - bm_sharpe, 3),
                "sharpe_improvement_qqq": round(float(metrics_net["Sharpe Ratio"]) - qqq_sharpe, 3),
                "final_equity_multiple": round(float(res_net["portfolio_value"].iloc[-1]) / float(benchmark.iloc[-1]), 2),
                "final_equity_multiple_qqq": round(float(res_net["portfolio_value"].iloc[-1]) / float(benchmark_qqq.iloc[-1]), 2),
            },
        },
        "annual_returns": annual_returns,
        "monthly_series": {
            "dates": monthly_dates,
            "strategy_equity": monthly_fund_val,
            "spy_equity": monthly_spy_val,
            "qqq_equity": monthly_qqq_val,
            "strategy_drawdown": monthly_fund_dd,
            "spy_drawdown": monthly_spy_dd,
            "qqq_drawdown": monthly_qqq_dd,
        },
        "daily_series": {
            "dates": daily_dates,
            "strategy_equity": daily_fund_val,
            "spy_equity": daily_spy_val,
            "qqq_equity": daily_qqq_val,
            "strategy_drawdown": daily_fund_dd,
            "spy_drawdown": daily_spy_dd,
            "qqq_drawdown": daily_qqq_dd,
        },
        "yearly_series": {
            "dates": yearly_dates,
            "strategy_equity": yearly_fund_val,
            "spy_equity": yearly_spy_val,
            "qqq_equity": yearly_qqq_val,
            "strategy_drawdown": yearly_fund_dd,
            "spy_drawdown": yearly_spy_dd,
            "qqq_drawdown": yearly_qqq_dd,
        },
    }

    print(f"[4/4] Writing output to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    sz_kb = os.path.getsize(output_path) / 1024
    print(f"[OK] Backtest results exported successfully -> {output_path} ({sz_kb:.1f} KB)")
    return payload


if __name__ == "__main__":
    generate_backtest_payload()
