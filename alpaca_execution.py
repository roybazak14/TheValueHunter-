"""
alpaca_execution.py — Alpaca Paper Trading Execution for The Value Hunter.

Reads `current_portfolio_signals.csv`, compares target allocations against
current Alpaca positions, and executes rebalancing orders.

Features:
    • Reads API keys from environment variables or .env file.
    • Evaluates liquidation equity from Alpaca.
    • Computes order deltas (sells first to free cash, then buys).
    • Supports --dry-run mode for pre-trade auditing without placing real orders.
    • Logs all executions to `alpaca_trade_log.csv`.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import yfinance as yf

# Load .env file with multi-path resolution
try:
    from dotenv import load_dotenv, find_dotenv

    # 1. Search starting from current working directory
    # 2. Search starting from script directory
    # 3. Check workspace parent directories
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cwd = os.getcwd()

    env_paths_to_check = [
        find_dotenv(usecwd=True),
        os.path.join(_cwd, ".env"),
        os.path.join(_script_dir, ".env"),
        os.path.join(os.path.dirname(_script_dir), ".env"),
        os.path.join(_script_dir, ".env.example"),
    ]

    loaded_env = False
    for p in env_paths_to_check:
        if p and os.path.isfile(p):
            load_dotenv(dotenv_path=p, override=True)
            loaded_env = True
            break
except ImportError:
    pass

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_ENDPOINT,
    ALPACA_PAPER,
    BOND_TICKER,
    DEFENSIVE_BOND_ALLOC,
)

logger = logging.getLogger("alpaca_exec")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ── Credential Resolution ─────────────────────────────────────────────────────

def get_alpaca_credentials() -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Retrieve Alpaca API keys, custom endpoint, and paper endpoint flag.
    Returns: (api_key, secret_key, endpoint_url, is_paper)
    """
    api_key = os.getenv("ALPACA_API_KEY") or ALPACA_API_KEY
    secret_key = os.getenv("ALPACA_SECRET_KEY") or ALPACA_SECRET_KEY
    endpoint = os.getenv("ALPACA_ENDPOINT") or getattr(sys.modules["config"], "ALPACA_ENDPOINT", None)

    paper_env = os.getenv("ALPACA_PAPER")
    if paper_env is not None:
        paper = paper_env.lower() in ("true", "1", "yes")
    elif endpoint and "paper" in endpoint.lower():
        paper = True
    else:
        paper = ALPACA_PAPER

    # Clean strings if present
    if api_key:
        api_key = str(api_key).strip()
    if secret_key:
        secret_key = str(secret_key).strip()
    if endpoint:
        endpoint = str(endpoint).strip()

    return api_key, secret_key, endpoint, paper


def init_trading_client(dry_run: bool = False):
    """Initialize Alpaca TradingClient or return None in dry-run with dummy keys."""
    api_key, secret_key, endpoint, paper = get_alpaca_credentials()

    if not api_key or not secret_key or "your_" in api_key.lower():
        if dry_run:
            logger.warning("No live Alpaca credentials provided. Running in simulated dry-run mode.")
            return None
        raise ValueError(
            "Alpaca credentials missing or placeholder! Please set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in your environment, .env file, or config.py."
        )

    from alpaca.trading.client import TradingClient
    client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        url_override=endpoint,
    )
    return client


# ── Market Price Helper ───────────────────────────────────────────────────────

def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch recent market prices for ticker list via yfinance fallback."""
    if not tickers:
        return {}
    prices: Dict[str, float] = {}
    try:
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        prices[t] = float(s.iloc[-1])
        else:
            s = data["Close"].dropna()
            if not s.empty and tickers:
                prices[tickers[0]] = float(s.iloc[-1])
    except Exception as e:
        logger.warning("Error fetching batch prices: %s", e)

    # Fallback individual fetch for any missing
    for t in tickers:
        if t not in prices or prices[t] <= 0:
            try:
                hist = yf.Ticker(t).history(period="5d")
                if not hist.empty:
                    prices[t] = float(hist["Close"].iloc[-1])
            except Exception:
                pass

    return prices


# ── Target Portfolio Reader ───────────────────────────────────────────────────

def load_target_portfolio(signals_csv: str) -> Tuple[Dict[str, float], str, str]:
    """
    Read `current_portfolio_signals.csv` and return target weights dict,
    signal date, and regime.
    """
    if not os.path.isabs(signals_csv):
        signals_csv = os.path.join(os.path.dirname(__file__), signals_csv)

    if not os.path.exists(signals_csv):
        raise FileNotFoundError(
            f"Signals file not found at '{signals_csv}'. Please run get_current_signals.py first!"
        )

    df = pd.read_csv(signals_csv)
    target_weights: Dict[str, float] = {}

    regime = df["regime"].iloc[0] if "regime" in df.columns else "Risk-ON"
    signal_date = df["signal_date"].iloc[0] if "signal_date" in df.columns else "Unknown"

    # Equities
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        w = float(row["target_weight"])
        target_weights[ticker] = w

    # Defensive bond allocation if Risk-OFF
    if "risk-off" in regime.lower():
        bond_tk = df["bond_ticker"].iloc[0] if ("bond_ticker" in df.columns and pd.notna(df["bond_ticker"].iloc[0])) else BOND_TICKER
        bond_w = float(df["bond_weight"].iloc[0]) if "bond_weight" in df.columns else DEFENSIVE_BOND_ALLOC
        target_weights[bond_tk] = bond_w

    return target_weights, signal_date, regime


# ── Rebalancing Calculator ────────────────────────────────────────────────────

def compute_orders(
    target_weights: Dict[str, float],
    current_positions: Dict[str, float],  # ticker -> current shares
    prices: Dict[str, float],
    equity: float,
    min_order_dollars: float = 1.0,
) -> Tuple[List[dict], List[dict]]:
    """
    Compute sell and buy orders required to reach target weights.
    Returns: (sell_orders, buy_orders)
    """
    sell_orders: List[dict] = []
    buy_orders: List[dict] = []

    # Union of all tickers in target + current holdings
    all_tickers = sorted(set(list(target_weights.keys()) + list(current_positions.keys())))

    for ticker in all_tickers:
        cur_shares = current_positions.get(ticker, 0.0)
        price = prices.get(ticker, 0.0)
        if price <= 0:
            logger.warning("Price unavailable for %s. Skipping order calculation.", ticker)
            continue

        cur_val = cur_shares * price
        cur_w = cur_val / equity if equity > 0 else 0.0

        target_w = target_weights.get(ticker, 0.0)
        target_val = equity * target_w
        target_shares = target_val / price

        delta_val = target_val - cur_val
        delta_shares = target_shares - cur_shares

        order_info = {
            "ticker": ticker,
            "price": price,
            "cur_shares": cur_shares,
            "cur_val": cur_val,
            "cur_weight": cur_w,
            "target_shares": target_shares,
            "target_val": target_val,
            "target_weight": target_w,
            "delta_shares": delta_shares,
            "delta_val": delta_val,
        }

        if abs(delta_val) < min_order_dollars:
            continue

        if delta_val < 0:
            # Sell (or full liquidation if target_w == 0)
            order_info["side"] = "sell"
            order_info["is_full_liquidation"] = (target_w == 0.0)
            order_info["order_shares"] = cur_shares if target_w == 0.0 else abs(delta_shares)
            order_info["order_notional"] = cur_val if target_w == 0.0 else abs(delta_val)
            sell_orders.append(order_info)
        elif delta_val > 0:
            # Buy
            order_info["side"] = "buy"
            order_info["is_full_liquidation"] = False
            order_info["order_shares"] = delta_shares
            order_info["order_notional"] = delta_val
            buy_orders.append(order_info)

    # Sort sells by largest notional first (free cash earliest)
    sell_orders.sort(key=lambda x: x["order_notional"], reverse=True)
    # Sort buys by largest notional first
    buy_orders.sort(key=lambda x: x["order_notional"], reverse=True)

    return sell_orders, buy_orders


# ── Execution Logic ───────────────────────────────────────────────────────────

def execute_orders_alpaca(
    trading_client,
    sell_orders: List[dict],
    buy_orders: List[dict],
    use_notional: bool = True,
    dry_run: bool = False,
) -> List[dict]:
    """Execute sell orders first, then buy orders, recording results."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    executed_log: List[dict] = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    order_queue = [("SELL", sell_orders), ("BUY", buy_orders)]

    for phase_name, orders in order_queue:
        if not orders:
            continue
        print(f"\n--- Processing {phase_name} Orders ({len(orders)}) ---")
        for ord_info in orders:
            ticker = ord_info["ticker"]
            side_str = ord_info["side"].upper()
            notional = round(ord_info["order_notional"], 2)
            shares = round(ord_info["order_shares"], 4)

            log_entry = {
                "timestamp": timestamp_str,
                "ticker": ticker,
                "side": side_str,
                "shares": shares,
                "notional": notional,
                "estimated_price": ord_info["price"],
                "status": "DRY_RUN" if dry_run else "PENDING",
                "order_id": "DRY_RUN_SIM" if dry_run else None,
                "error": None,
            }

            if dry_run or trading_client is None:
                print(f"  [DRY-RUN] {side_str:<4} {ticker:<6} | {shares:>10.4f} shs | ~${notional:>9,.2f}")
                executed_log.append(log_entry)
                continue

            try:
                side_enum = OrderSide.SELL if side_str == "SELL" else OrderSide.BUY

                # For full position liquidations (target allocation = 0%), do NOT calculate quantity from price.
                # Always use client.close_position(symbol) to cleanly liquidate 100% of held shares without rounding errors.
                if side_str == "SELL" and ord_info.get("is_full_liquidation", False):
                    try:
                        alpaca_order = trading_client.close_position(symbol_or_asset_id=ticker)
                        order_id = getattr(alpaca_order, "id", None) or "CLOSED"
                        status = getattr(alpaca_order, "status", None) or "SUBMITTED"
                        log_entry["status"] = str(status)
                        log_entry["order_id"] = str(order_id)
                        print(f"  [LIQUIDATED] SELL {ticker:<6} | 100% Position via close_position() | ID: {order_id}")
                    except Exception as close_exc:
                        if "position does not exist" in str(close_exc).lower() or "404" in str(close_exc):
                            log_entry["status"] = "ALREADY_CLOSED"
                            print(f"  [ALREADY CLOSED] {ticker:<6} position does not exist")
                        else:
                            raise close_exc
                elif side_str == "SELL":
                    # For partial sells, explicitly use exact shares (capped to cur_shares) to prevent over-selling
                    exact_qty = min(round(ord_info["order_shares"], 4), ord_info.get("cur_shares", shares))
                    order_req = MarketOrderRequest(
                        symbol=ticker,
                        qty=exact_qty,
                        side=side_enum,
                        time_in_force=TimeInForce.DAY,
                    )
                    alpaca_order = trading_client.submit_order(order_req)
                    log_entry["status"] = str(alpaca_order.status)
                    log_entry["order_id"] = str(alpaca_order.id)
                    print(f"  [SUBMITTED] SELL {ticker:<6} | {exact_qty} shs (~${notional:,.2f}) | ID: {alpaca_order.id}")
                else:
                    # Buy orders: use notional (dollar amount) or qty (shares)
                    if use_notional:
                        order_req = MarketOrderRequest(
                            symbol=ticker,
                            notional=notional,
                            side=side_enum,
                            time_in_force=TimeInForce.DAY,
                        )
                    else:
                        order_req = MarketOrderRequest(
                            symbol=ticker,
                            qty=shares,
                            side=side_enum,
                            time_in_force=TimeInForce.DAY,
                        )

                    alpaca_order = trading_client.submit_order(order_req)
                    log_entry["status"] = str(alpaca_order.status)
                    log_entry["order_id"] = str(alpaca_order.id)
                    print(f"  [SUBMITTED] BUY  {ticker:<6} | ${notional:,.2f} | ID: {alpaca_order.id}")
            except Exception as exc:
                log_entry["status"] = "FAILED"
                log_entry["error"] = str(exc)
                logger.error("Failed to place %s order for %s: %s", side_str, ticker, exc)

            executed_log.append(log_entry)

    return executed_log


# ── Audit Log Writer ──────────────────────────────────────────────────────────

def save_trade_log(executed_log: List[dict], log_path: str = "alpaca_trade_log.csv"):
    """Append executed order report to CSV log."""
    if not executed_log:
        return
    if not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(__file__), log_path)

    df_new = pd.DataFrame(executed_log)
    if os.path.exists(log_path):
        try:
            df_old = pd.read_csv(log_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(log_path, index=False)
    print(f"\n  [Audit Log] Execution report updated -> {log_path}")


# ── Main Runner ───────────────────────────────────────────────────────────────

def run_alpaca_rebalance(
    signals_csv: str = "current_portfolio_signals.csv",
    dry_run: bool = False,
    simulated_equity: float = 100_000.0,
    log_csv: str = "alpaca_trade_log.csv",
):
    print("=" * 76)
    print("  THE VALUE HUNTER — ALPACA PAPER TRADING REBALANCER")
    print("=" * 76)
    mode_str = "DRY-RUN (Simulated / No Orders Placed)" if dry_run else "LIVE PAPER TRADING"
    print(f"  Mode           : {mode_str}")

    # 1. Load target signals
    target_weights, signal_date, regime = load_target_portfolio(signals_csv)
    print(f"  Signal Date    : {signal_date}")
    print(f"  Macro Regime   : {regime}")
    print(f"  Target Universe: {len(target_weights)} assets ({sum(target_weights.values()) * 100:.0f}% allocated)")

    # 2. Connect to Alpaca
    trading_client = None
    account_equity = simulated_equity
    current_positions: Dict[str, float] = {}

    try:
        trading_client = init_trading_client(dry_run=dry_run)
        if trading_client is not None:
            account = trading_client.get_account()
            account_equity = float(account.equity)
            print(f"  Alpaca Status  : Account {account.account_number} ({account.status})")
            print(f"  Alpaca Equity  : ${account_equity:,.2f} (Buying Power: ${float(account.buying_power):,.2f})")

            # Fetch open positions
            positions = trading_client.get_all_positions()
            for p in positions:
                current_positions[p.symbol.upper()] = float(p.qty)
            print(f"  Open Positions : {len(current_positions)} positions currently held")
        else:
            print(f"  Simulated Equity: ${account_equity:,.2f} (Default Dry-Run Baseline)")
    except Exception as exc:
        if dry_run:
            logger.warning("Could not connect to Alpaca: %s. Proceeding with simulated portfolio.", exc)
            print(f"  Simulated Equity: ${account_equity:,.2f} (Fallback Baseline)")
        else:
            raise exc

    # 3. Market prices
    all_needed_tickers = list(set(list(target_weights.keys()) + list(current_positions.keys())))
    print(f"\n[1/3] Fetching latest market prices for {len(all_needed_tickers)} tickers ...")
    prices = fetch_current_prices(all_needed_tickers)
    print(f"      Retrieved prices for {len(prices)} / {len(all_needed_tickers)} symbols")

    # 4. Compute order deltas
    print("\n[2/3] Calculating rebalancing deltas ...")
    sell_orders, buy_orders = compute_orders(
        target_weights=target_weights,
        current_positions=current_positions,
        prices=prices,
        equity=account_equity,
    )

    total_sell_val = sum(o["order_notional"] for o in sell_orders)
    total_buy_val  = sum(o["order_notional"] for o in buy_orders)
    print(f"      Planned Sells: {len(sell_orders):>2} orders totaling ${total_sell_val:>10,.2f}")
    print(f"      Planned Buys : {len(buy_orders):>2} orders totaling ${total_buy_val:>10,.2f}")

    # 5. Display planned summary table
    print("\n" + "-" * 76)
    print(f"{'Side':<6} {'Ticker':<8} {'Cur %':>7} {'Tgt %':>7} {'Price':>10} {'Est Shares':>12} {'Est Notional':>14}")
    print("-" * 76)

    for o in sell_orders:
        print(f"SELL   {o['ticker']:<8} {o['cur_weight']*100:>6.1f}% {o['target_weight']*100:>6.1f}% ${o['price']:>9.2f} {o['order_shares']:>12.4f} ${o['order_notional']:>13,.2f}")
    for o in buy_orders:
        print(f"BUY    {o['ticker']:<8} {o['cur_weight']*100:>6.1f}% {o['target_weight']*100:>6.1f}% ${o['price']:>9.2f} {o['order_shares']:>12.4f} ${o['order_notional']:>13,.2f}")
    print("-" * 76)

    # 6. Execution
    print("\n[3/3] Executing orders ...")
    executed_log = execute_orders_alpaca(
        trading_client=trading_client,
        sell_orders=sell_orders,
        buy_orders=buy_orders,
        use_notional=True,
        dry_run=dry_run,
    )

    # 7. Save Audit Log
    save_trade_log(executed_log, log_path=log_csv)
    print("=" * 76)


def parse_args():
    p = argparse.ArgumentParser(description="Alpaca Paper Trading Execution for The Value Hunter")
    p.add_argument("--dry-run", action="store_true", help="Preview planned orders without submitting to Alpaca")
    p.add_argument("--signals", default="current_portfolio_signals.csv", help="Path to portfolio signals CSV")
    p.add_argument("--equity", type=float, default=100_000.0, help="Simulated equity in dry-run mode (default $100k)")
    p.add_argument("--log", default="alpaca_trade_log.csv", help="Execution audit log CSV output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_alpaca_rebalance(
        signals_csv=args.signals,
        dry_run=args.dry_run,
        simulated_equity=args.equity,
        log_csv=args.log,
    )

