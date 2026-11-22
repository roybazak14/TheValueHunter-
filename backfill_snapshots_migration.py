"""
backfill_snapshots_migration.py — One-Time Backfill Migration Script.

Repairs missing trading snapshots in `fund_users.db` between 2026-10-26 and 2026-11-01
for the Top-10 equal-weight portfolio:
Holdings: CVX, HON, MTCH, ALGN, ROKU, NXPI, QCOM, REGN, BAC, ON
Benchmarks: SPY, QQQ
"""

import os
import sys
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime

# Script directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.getenv("VH_DATA_DIR", BASE_DIR))
DB_PATH = os.path.abspath(os.getenv("VH_DB_PATH", os.path.join(DATA_DIR, "fund_users.db")))
CSV_PATH = os.path.join(DATA_DIR, "live_portfolio_history.csv")

# Baseline constants
INITIAL_CAPITAL = 100_000.00
SPY_INCEPTION_PRICE = 773.09
QQQ_INCEPTION_PRICE = 725.20

TOP10_TICKERS = ["CVX", "HON", "MTCH", "ALGN", "ROKU", "NXPI", "QCOM", "REGN", "BAC", "ON"]
BENCHMARK_TICKERS = ["SPY", "QQQ"]
TARGET_DATES = ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_database_snapshots(conn: sqlite3.Connection):
    """Inspect and print current snapshots in database."""
    print("=" * 80)
    print("STEP 1: INSPECTING EXISTING SNAPSHOTS IN fund_users.db")
    print("=" * 80)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            date TEXT PRIMARY KEY,
            portfolio_value REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            spy_price REAL NOT NULL,
            qqq_price REAL NOT NULL,
            fund_return_pct REAL NOT NULL,
            spy_return_pct REAL NOT NULL,
            qqq_return_pct REAL NOT NULL,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'live'
        )
        """
    )
    conn.commit()

    cur.execute("SELECT date, portfolio_value, fund_return_pct, spy_return_pct, qqq_return_pct, source FROM portfolio_snapshots ORDER BY date ASC")
    rows = cur.fetchall()
    print(f"Total existing snapshot records: {len(rows)}")
    for r in rows:
        print(f"  [{r['date']}] Equity: ${r['portfolio_value']:>10,.2f} | Fund: {r['fund_return_pct']:>+6.2f}% | SPY: {r['spy_return_pct']:>+6.2f}% | QQQ: {r['qqq_return_pct']:>+6.2f}% ({r['source']})")

    existing_dates = set(r["date"] for r in rows)
    return existing_dates


def fetch_historical_prices():
    """Fetch official adjusted closing prices via yfinance."""
    print("\n" + "=" * 80)
    print("STEP 2: FETCHING OFFICIAL ADJUSTED CLOSING PRICES VIA YFINANCE")
    print("=" * 80)
    all_syms = TOP10_TICKERS + BENCHMARK_TICKERS
    print(f"Querying symbols: {', '.join(all_syms)}")
    print(f"Period: 2026-10-24 to 2026-11-01 (including 2026-10-26 baseline)")

    df = yf.download(all_syms, start="2026-10-24", end="2026-11-01", auto_adjust=True, progress=False)
    close_df = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df

    print("\nClosing Prices Table:")
    print(close_df.to_string())
    return close_df


def compute_and_insert_backfill(conn: sqlite3.Connection, close_df: pd.DataFrame):
    """Calculate daily equal-weight values and insert snapshots."""
    print("\n" + "=" * 80)
    print("STEP 3: COMPUTING EQUAL-WEIGHT NAV AND MIGRATING SNAPSHOTS")
    print("=" * 80)

    # Retrieve base portfolio value at 2026-10-26
    cur = conn.cursor()
    cur.execute("SELECT portfolio_value, cash FROM portfolio_snapshots WHERE date = '2026-10-26'")
    base_row = cur.fetchone()
    if base_row:
        v_prev = float(base_row["portfolio_value"])
        base_cash = float(base_row["cash"])
        print(f"Anchor date 2026-10-26 found: Portfolio Value = ${v_prev:,.2f}, Cash = ${base_cash:,.2f}")
    else:
        v_prev = 100_372.01
        base_cash = 0.19
        print(f"Anchor date 2026-10-26 not found in DB. Using standard anchor value: ${v_prev:,.2f}")

    # Compute daily percentage change for each constituent
    pct_changes = close_df[TOP10_TICKERS].pct_change()

    inserted_records = []
    for d in TARGET_DATES:
        matching_dt = None
        for dt_idx in close_df.index:
            if dt_idx.strftime("%Y-%m-%d") == d:
                matching_dt = dt_idx
                break

        if matching_dt is None:
            print(f"Warning: No market data for date {d}. Skipping.")
            continue

        # Daily equal-weight portfolio return across the 10 holdings
        day_returns = pct_changes.loc[matching_dt]
        d_ret = float(day_returns.mean())
        d_ret_pct = d_ret * 100.0

        # Calculate new portfolio value
        v_current = round(v_prev * (1.0 + d_ret), 2)
        pos_val = round(v_current - base_cash, 2)
        cum_fund_ret = round(((v_current / INITIAL_CAPITAL) - 1.0) * 100.0, 4)

        # Benchmark prices & returns
        spy_px = float(close_df.loc[matching_dt, "SPY"])
        qqq_px = float(close_df.loc[matching_dt, "QQQ"])
        spy_cum_ret = round(((spy_px / SPY_INCEPTION_PRICE) - 1.0) * 100.0, 4)
        qqq_cum_ret = round(((qqq_px / QQQ_INCEPTION_PRICE) - 1.0) * 100.0, 4)

        timestamp_str = f"{d} 16:00:00 -04:00 (Top10 EW Backfill)"

        # Insert or replace record in SQLite
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_snapshots
                (date, portfolio_value, cash, positions_value, spy_price, qqq_price, fund_return_pct, spy_return_pct, qqq_return_pct, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'top10_ew_backfill')
                """,
                (
                    d,
                    v_current,
                    base_cash,
                    pos_val,
                    round(spy_px, 4),
                    round(qqq_px, 4),
                    cum_fund_ret,
                    spy_cum_ret,
                    qqq_cum_ret,
                    timestamp_str,
                ),
            )

        print(f"Inserted [{d}]: Value = ${v_current:>10,.2f} | 1D: {d_ret_pct:>+6.2f}% | Cum: {cum_fund_ret:>+6.2f}% | SPY: ${spy_px:>6.2f} ({spy_cum_ret:>+6.2f}%) | QQQ: ${qqq_px:>6.2f} ({qqq_cum_ret:>+6.2f}%)")
        inserted_records.append(d)
        v_prev = v_current

    # Export synchronized data to live_portfolio_history.csv
    cur.execute(
        """
        SELECT date, portfolio_value, cash, positions_value, spy_price, qqq_price,
               fund_return_pct, spy_return_pct, qqq_return_pct, updated_at
        FROM portfolio_snapshots
        ORDER BY date ASC
        """
    )
    all_rows = [dict(r) for r in cur.fetchall()]
    df_export = pd.DataFrame(all_rows)
    df_export.to_csv(CSV_PATH, index=False)
    print(f"\nSynchronized CSV exported to -> {CSV_PATH} ({len(df_export)} records)")


def verify_database_continuity(conn: sqlite3.Connection):
    """Verify that there are no gaps between Inception and latest date."""
    print("\n" + "=" * 80)
    print("STEP 4: VERIFYING DATABASE CHRONOLOGICAL CONTINUITY")
    print("=" * 80)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, portfolio_value, fund_return_pct, spy_price, spy_return_pct, qqq_price, qqq_return_pct, source, updated_at
        FROM portfolio_snapshots
        ORDER BY date ASC
        """
    )
    rows = cur.fetchall()

    print(f"{'Date':<12} | {'Equity':<14} | {'Fund Ret':<10} | {'SPY Price':<10} | {'SPY Ret':<10} | {'QQQ Ret':<10} | {'Source':<20}")
    print("-" * 96)
    for r in rows:
        print(f"{r['date']:<12} | ${r['portfolio_value']:>12,.2f} | {r['fund_return_pct']:>+8.2f}% | ${r['spy_price']:>8.2f} | {r['spy_return_pct']:>+8.2f}% | {r['qqq_return_pct']:>+8.2f}% | {r['source']:<20}")

    dates = [r["date"] for r in rows]
    print("-" * 96)
    print(f"Chronological Dates: {' -> '.join(dates)}")

    # Check for expected trading days
    expected_days = ["2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]
    missing = [d for d in expected_days if d not in dates]
    if not missing:
        print("\nSUCCESS: All trading days from Inception to 2026-10-30 are fully present with zero gaps!")
    else:
        print(f"\nWARNING: Missing dates detected: {missing}")


def main():
    conn = get_db_connection()
    try:
        inspect_database_snapshots(conn)
        close_df = fetch_historical_prices()
        compute_and_insert_backfill(conn, close_df)
        verify_database_continuity(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

