"""
backtest_engine.py — Event-driven monthly backtest with no look-ahead bias.

Execution model:
    • Signals generated on LAST TRADING DAY of month M  (close prices).
    • Trades executed at OPEN of FIRST TRADING DAY of month M+1.
    • Portfolio value updated daily using adjusted-close prices.
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TRADING_DAYS_YEAR,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
)

logger = logging.getLogger(__name__)


# ── Core engine ───────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Pure-Python monthly rebalancing backtest.

    Parameters
    ----------
    prices_close : pd.DataFrame
        Daily adjusted-close prices for ALL tickers (equities + SPY + TLT + SHV).
    prices_open  : pd.DataFrame
        Daily open prices for ALL tickers (used for execution).
    signals      : list[dict]
        Output of strategy.generate_monthly_signals().
    initial_capital : float
    """

    def __init__(
        self,
        prices_close: pd.DataFrame,
        prices_open: pd.DataFrame,
        signals: List[dict],
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        cash_buffer: float = 0.0,
    ):
        self.close   = prices_close.copy()
        self.opens   = prices_open.copy()
        self.signals = sorted(signals, key=lambda x: x["signal_date"])
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate   = slippage_rate
        self.cash_buffer     = cash_buffer

        # Align both DataFrames on the same date index
        common_idx = self.close.index.intersection(self.opens.index)
        self.close  = self.close.loc[common_idx]
        self.opens  = self.opens.loc[common_idx]

        # Build a fast signal lookup: signal_date → signal dict
        self._signal_map: Dict[pd.Timestamp, dict] = {
            s["signal_date"]: s for s in self.signals
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_first_trading_day_of_month(self, year: int, month: int) -> pd.Timestamp | None:
        """Return the first trading day of (year, month) in the price index."""
        mask = (self.close.index.year == year) & (self.close.index.month == month)
        days = self.close.index[mask]
        return days[0] if len(days) > 0 else None

    def _execution_price(self, date: pd.Timestamp, ticker: str) -> float | None:
        """
        Get open price for ticker on date.
        Falls back to close if open is NaN.
        """
        if ticker not in self.opens.columns:
            return None
        try:
            op = self.opens.loc[date, ticker]
            if pd.isna(op) or op <= 0:
                # fall back to close
                cl = self.close.loc[date, ticker]
                return float(cl) if not pd.isna(cl) and cl > 0 else None
            return float(op)
        except KeyError:
            return None

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest and return a daily portfolio value Series.

        Returns
        -------
        pd.DataFrame with columns:
            portfolio_value, daily_return, drawdown
        """
        all_dates = self.close.index
        portfolio_value = np.full(len(all_dates), np.nan)
        portfolio_value[0] = self.capital

        # Current holdings: dict { ticker → (shares, weight_target) }
        holdings: Dict[str, float] = {}   # ticker → shares
        cash = self.capital

        # Build set of month-end dates from signals (signal_date)
        signal_dates = set(self._signal_map.keys())

        # Find month-start execution days: for each signal, find next month's 1st day
        execution_map: Dict[pd.Timestamp, dict] = {}
        for sig in self.signals:
            sd = sig["signal_date"]
            # Next month
            next_month = sd + pd.offsets.MonthBegin(1)
            year, month = next_month.year, next_month.month
            exec_date = self._get_first_trading_day_of_month(year, month)
            if exec_date is not None:
                execution_map[exec_date] = sig

        logger.info(
            "Backtest period: %s → %s  |  %d rebalances mapped.",
            all_dates[0].date(), all_dates[-1].date(), len(execution_map),
        )

        for i, date in enumerate(all_dates):
            if i == 0:
                portfolio_value[0] = cash
                continue

            # ── Rebalance if this is an execution day ──────────────────────
            if date in execution_map:
                sig = execution_map[date]
                eq_tickers  = sig["equity_tickers"]
                eq_weight   = sig["equity_weight"]
                bond_weight = sig["bond_weight"]
                bond_ticker = sig["bond_ticker"]

                # Current portfolio value (before rebalance)
                total_val = cash
                for ticker, shares in holdings.items():
                    cl = self.close.loc[date, ticker] if ticker in self.close.columns else np.nan
                    if not np.isnan(cl):
                        total_val += shares * cl

                # Build target allocation (after reserving cash buffer)
                investable_val = total_val * (1.0 - self.cash_buffer)
                target: Dict[str, float] = {}   # ticker → target $ value
                for tk in eq_tickers:
                    target[tk] = investable_val * eq_weight
                if bond_weight > 0 and bond_ticker in self.close.columns:
                    target[bond_ticker] = investable_val * bond_weight

                # Liquidate positions not in target (sell first, buy after)
                new_holdings: Dict[str, float] = {}
                for ticker, shares in holdings.items():
                    if ticker not in target:
                        raw_price = self._execution_price(date, ticker)
                        if raw_price and shares > 0:
                            # Sell price affected by slippage downwards
                            exec_price = raw_price * (1.0 - self.slippage_rate)
                            gross_proceeds = shares * exec_price
                            comm = gross_proceeds * self.commission_rate
                            cash += (gross_proceeds - comm)
                    else:
                        new_holdings[ticker] = shares

                # Buy / adjust positions in target
                for ticker, dollar_target in target.items():
                    raw_price = self._execution_price(date, ticker)
                    if raw_price is None or raw_price <= 0:
                        continue
                    current_shares = new_holdings.get(ticker, 0.0)
                    current_val    = current_shares * raw_price
                    delta_val      = dollar_target - current_val

                    if delta_val > 0:
                        # Buy: slippage moves execution price upwards
                        buy_price = raw_price * (1.0 + self.slippage_rate)
                        # Cash outlay includes buy commission: delta_shares * buy_price * (1 + comm) = delta_val
                        eff_cost_per_share = buy_price * (1.0 + self.commission_rate)
                        delta_shares = delta_val / eff_cost_per_share
                        total_cost = delta_shares * eff_cost_per_share

                        new_holdings[ticker] = current_shares + delta_shares
                        cash -= total_cost
                    elif delta_val < 0:
                        # Trim position: sell price moves downwards with slippage
                        sell_price = raw_price * (1.0 - self.slippage_rate)
                        shares_to_sell = min(current_shares, abs(delta_val) / raw_price)
                        gross_proceeds = shares_to_sell * sell_price
                        comm = gross_proceeds * self.commission_rate

                        rem_shares = current_shares - shares_to_sell
                        if rem_shares > 1e-6:
                            new_holdings[ticker] = rem_shares
                        elif ticker in new_holdings:
                            del new_holdings[ticker]
                        cash += (gross_proceeds - comm)

                holdings = new_holdings

            # ── Mark-to-market ────────────────────────────────────────────
            total = cash
            for ticker, shares in holdings.items():
                if ticker in self.close.columns:
                    price = self.close.loc[date, ticker]
                    if not np.isnan(price):
                        total += shares * price
            portfolio_value[i] = total

        # ── Build results DataFrame ────────────────────────────────────────
        pv = pd.Series(portfolio_value, index=all_dates, name="portfolio_value")
        pv = pv.ffill().dropna()

        daily_ret = pv.pct_change().fillna(0.0)
        roll_max  = pv.cummax()
        drawdown  = (pv - roll_max) / roll_max

        results = pd.DataFrame({
            "portfolio_value": pv,
            "daily_return":    daily_ret,
            "drawdown":        drawdown,
        })
        return results


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """Compute standard performance metrics from a results DataFrame."""
    pv  = results["portfolio_value"]
    ret = results["daily_return"]
    dd  = results["drawdown"]

    n_years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = (pv.iloc[-1] / initial_capital) ** (1 / n_years) - 1

    ann_vol = ret.std() * np.sqrt(TRADING_DAYS_YEAR)

    # Sharpe (risk-free ≈ 0 for simplicity; could subtract TBill rate)
    sharpe = (ret.mean() * TRADING_DAYS_YEAR) / (ann_vol + 1e-12)

    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Best / worst year
    annual = (1 + ret).resample("YE").prod() - 1
    best_year  = annual.max()
    worst_year = annual.min()

    total_return = (pv.iloc[-1] / initial_capital) - 1

    return {
        "CAGR (%)":           round(cagr * 100, 2),
        "Total Return (%)":   round(total_return * 100, 2),
        "Annual Vol (%)":     round(ann_vol * 100, 2),
        "Sharpe Ratio":       round(sharpe, 3),
        "Max Drawdown (%)":   round(max_dd * 100, 2),
        "Calmar Ratio":       round(calmar, 3),
        "Best Year (%)":      round(best_year * 100, 2),
        "Worst Year (%)":     round(worst_year * 100, 2),
        "Start":              str(pv.index[0].date()),
        "End":                str(pv.index[-1].date()),
        "Years":              round(n_years, 2),
    }

