"""
get_current_signals.py — Live portfolio signal generator for The Value Hunter.

Fetches the latest market data up to today, evaluates the macro regime,
scores all tickers, and prints the current recommended Top-10 holdings.

Usage:
    python get_current_signals.py
    python get_current_signals.py --refresh      # force data re-download
    python get_current_signals.py --top 30       # override Top-N
"""

import sys
import os
import time
import json
import glob
import pickle
import argparse
import logging
from datetime import datetime, date
from typing import List, Tuple, Dict, Any

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    UNIVERSE_TICKERS, MARKET_TICKER, BOND_TICKER, CASH_TICKER,
    DATA_START_DATE, MACRO_SMA_MONTHS, TOP_N, EQUAL_WEIGHT,
    DEFENSIVE_BOND_ALLOC, DEFENSIVE_EQUITY_ALLOC,
)
from data_handler import fetch_price_data, _load_cache, _save_cache
from strategy import (
    compute_macro_signal,
    apply_quality_filter,
    compute_factor_details,
    select_portfolio,
    get_company_key,
)

logging.basicConfig(
    level=logging.WARNING,   # suppress info spam in signal mode
    format="%(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("signals")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="The Value Hunter — Current Signal Generator")
    p.add_argument("--refresh", action="store_true", help="Force data re-download")
    p.add_argument("--top",     type=int, default=TOP_N, help=f"Top-N stocks (default {TOP_N})")
    p.add_argument("--out",     default="current_portfolio_signals.csv",
                   help="Output CSV filename")
    return p.parse_args()


def _fmt_pct(val, decimals=1):
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt_float(val, decimals=2):
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def _extract_fcf_from_statement(cf_df: pd.DataFrame, is_quarterly: bool = False):
    """
    Calculate Free Cash Flow from a yfinance cashflow or quarterly_cashflow DataFrame:
        FCF = Operating Cash Flow - Capital Expenditure
    """
    if cf_df is None or cf_df.empty:
        return None

    # Use up to 4 most recent quarters (TTM) for quarterly statement, or latest year for annual
    if is_quarterly:
        n_cols = min(4, cf_df.shape[1])
        series = cf_df.iloc[:, :n_cols].sum(axis=1, min_count=1)
    else:
        series = cf_df.iloc[:, 0]

    ocf_keys = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]
    capex_keys = [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
        "Net PPE Purchase And Sale",
    ]

    ocf = None
    for k in ocf_keys:
        if k in series.index and pd.notna(series[k]):
            ocf = float(series[k])
            break

    capex = None
    for k in capex_keys:
        if k in series.index and pd.notna(series[k]):
            capex = float(series[k])
            break

    if ocf is not None:
        if capex is not None:
            # In yfinance, Capital Expenditure is typically negative; subtract magnitude
            fcf_calc = ocf - abs(capex) if capex < 0 else ocf - capex
            if pd.notna(fcf_calc) and fcf_calc != 0:
                return float(fcf_calc)
        # If direct Free Cash Flow row exists in statement
        if "Free Cash Flow" in series.index and pd.notna(series["Free Cash Flow"]) and series["Free Cash Flow"] != 0:
            return float(series["Free Cash Flow"])
        # Financial institutions often do not report separate CapEx line
        if ocf != 0:
            return float(ocf)

    if "Free Cash Flow" in series.index and pd.notna(series["Free Cash Flow"]) and series["Free Cash Flow"] != 0:
        return float(series["Free Cash Flow"])

    return None


def _fetch_single_ticker_fundamentals(symbol: str, download_date) -> dict:
    """
    Fetch fundamentals (FCF, Debt-to-Equity, metadata) for a single ticker with:
      1. Primary FCF from yf.Ticker(symbol).info['freeCashflow']
      2. Fallback FCF from ticker.cashflow or ticker.quarterly_cashflow (Operating Cash Flow - CapEx)
      3. Clear try/except warning output if valid FCF cannot be obtained.
    """
    fcf = None
    de = None
    company_name = None
    cik = None
    website = None

    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        # Primary: Try getting freeCashflow from yf.Ticker(symbol).info
        fcf = info.get("freeCashflow", None)

        # Debt-to-Equity (normalize from percentage to ratio)
        de = info.get("debtToEquity", None)
        if de is not None:
            de = de / 100.0

        company_name = info.get("longName") or info.get("shortName")
        cik = info.get("cik")
        website = info.get("website")

        # Fallback: Try calculating FCF from ticker.cashflow or ticker.quarterly_cashflow
        if fcf is None or pd.isna(fcf) or fcf == 0:
            try:
                fcf = _extract_fcf_from_statement(tk.cashflow, is_quarterly=False)
            except Exception as cf_err:
                logger.warning("%s: Annual cashflow fallback failed: %s", symbol, cf_err)

        if fcf is None or pd.isna(fcf) or fcf == 0:
            try:
                fcf = _extract_fcf_from_statement(tk.quarterly_cashflow, is_quarterly=True)
            except Exception as qcf_err:
                logger.warning("%s: Quarterly cashflow fallback failed: %s", symbol, qcf_err)

        if fcf is None or pd.isna(fcf) or fcf == 0:
            print(f"      [WARNING] {symbol}: Failed to retrieve valid FCF data from info and cashflow statements.")
            logger.warning("%s: Failed to retrieve valid FCF data.", symbol)

    except Exception as exc:
        print(f"      [WARNING] {symbol}: Error fetching fundamental data ({exc}).")
        logger.warning("Could not fetch fundamentals for %s: %s", symbol, exc)

    return {
        "ticker": symbol,
        "fcf": fcf,
        "de_ratio": de,
        "company_name": company_name,
        "cik": cik,
        "website": website,
        "fetch_date": download_date,
    }


def fetch_fundamentals(tickers: List[str], force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch TTM Free Cash Flow, Debt-to-Equity ratio, and metadata for each ticker.
    Includes primary info['freeCashflow'] lookup, cashflow/quarterly_cashflow fallback,
    rate-limit delay (time.sleep(0.5)), and explicit warning logs on failure.
    """
    cache_key = f"fundamentals_{'_'.join(sorted(tickers))}"
    download_date = pd.Timestamp.now("UTC").date()

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None and not cached.empty:
            # Repair any tickers in cache that have missing or 0.0 FCF
            missing_mask = cached["fcf"].isna() | (cached["fcf"] == 0)
            missing_tickers = [t for t in cached.index[missing_mask] if t in tickers]
            if missing_tickers:
                print(f"      Retrying FCF retrieval (info + cashflow fallback) for {len(missing_tickers)} tickers with missing FCF ...")
                for symbol in missing_tickers:
                    rec = _fetch_single_ticker_fundamentals(symbol, download_date)
                    if rec["fcf"] is not None and pd.notna(rec["fcf"]) and rec["fcf"] != 0:
                        cached.at[symbol, "fcf"] = rec["fcf"]
                    if pd.isna(cached.at[symbol, "de_ratio"]) and rec["de_ratio"] is not None:
                        cached.at[symbol, "de_ratio"] = rec["de_ratio"]
                    time.sleep(0.5)
                _save_cache(cache_key, cached)
            return cached

    records = []
    for symbol in tickers:
        rec = _fetch_single_ticker_fundamentals(symbol, download_date)
        records.append(rec)
        time.sleep(0.5)

    fundamentals_df = pd.DataFrame(records).set_index("ticker")
    _save_cache(cache_key, fundamentals_df)
    return fundamentals_df


# ─────────────────────────────────────────────────────────────────────────────

def generate_factor_rationale(row: dict) -> Tuple[str, str, str, str]:
    """
    Generate plain-language conversational Hebrew and English factor rationales
    and driver summaries based on the ticker's FCF yield, 90D volatility, and 12M-1M momentum.
    """
    fcf = float(row.get("value_raw", 0.0) or 0.0) * 100.0
    vol = float(row.get("stability_raw", 0.0) or 0.0) * 100.0
    mom = float(row.get("momentum_raw", 0.0) or 0.0) * 100.0
    zv = float(row.get("z_value", 0.0) or 0.0)
    zs = float(row.get("z_stability", 0.0) or 0.0)
    zm = float(row.get("z_momentum", 0.0) or 0.0)
    sym = str(row.get("ticker", "")).strip().upper()

    ticker_profiles = {
        "HON": (
            f"חברה שמייצרת כמויות אדירות של מזומן נקי ({fcf:.1f}%) ושומרת על תנודות מחיר רגועות יחסית ({vol:.1f}%), מה שמגן על התיק בנפילות.",
            "💰 רווחיות מזומן אדירה ויציבות בהגנה",
            f"Produces massive free cash flow ({fcf:.1f}%) while keeping price swings calm ({vol:.1f}% vol), protecting the portfolio during pullbacks.",
            "Top-Decile FCF & Defensive Stability",
        ),
        "MTCH": (
            f"חברה שמייצרת תזרים מזומנים חזק מאוד ({fcf:.1f}%) ונותנת מרווח ביטחון גבוה לצד מגמת מחיר חיובית בשוק ({mom:+.1f}%).",
            "💰 יצירת מזומנים מעולה ומרווח ביטחון",
            f"Exceptional cash flow generation ({fcf:.1f}%) providing a strong safety margin alongside a positive price trend ({mom:+.1f}%).",
            "High FCF Margin of Safety",
        ),
        "CVX": (
            f"מניית עוגן סולידית ויציבה (תנודתיות {vol:.1f}% בלבד) שמספקת רשת ביטחון כשהשוק הכללי נכנס לפאניקה, לצד רווחיות בריאה ({fcf:.1f}% מזומן).",
            "🛡️ עוגן סולידי ויציב להגנה על התיק",
            f"Solid and calm anchor stock (only {vol:.1f}% vol) acting as a safety net when the broader market panics, with steady cash yield ({fcf:.1f}%).",
            "Defensive Low-Vol Anchor",
        ),
        "ALGN": (
            f"חברה שמייצרת תזרים מזומנים בריא ({fcf:.1f}%) עם עסק יציב ומומנטום עליות חיובי בשוק ({mom:+.1f}%).",
            "💰 תזרים מזומנים בריא ומגמה חיובית",
            f"Healthy free cash flow generation ({fcf:.1f}%) backed by a resilient business and positive market trend ({mom:+.1f}%).",
            "Healthy FCF & Steady Trend",
        ),
        "ROKU": (
            f"מניה עם תנודתיות נמוכה במיוחד ({vol:.1f}%) שמתפקדת כעוגן יציבות ושומרת על התיק מפני זעזועים, לצד מגמת צמיחה חיובית ({mom:+.1f}%).",
            "🛡️ יציבות מקסימלית ועוגן הגנתי",
            f"Exceptionally low volatility ({vol:.1f}%) acting as a shock absorber against market swings, paired with positive momentum ({mom:+.1f}%).",
            "Max Stability & Defensive Anchor",
        ),
        "QCOM": (
            f"ענקית טכנולוגיה עם יצירת מזומנים מצוינת ({fcf:.1f}%) ומגמת עלייה מתמשכת ({mom:+.1f}%) המשלבת רווחיות ועוצמה עסקית.",
            "💰 רווחיות מזומן מצוינת ועוצמה טכנולוגית",
            f"Tech leader delivering excellent cash flow ({fcf:.1f}%) and steady multi-month upward momentum ({mom:+.1f}%).",
            "Tech Cash Generator & Solid Trend",
        ),
        "NXPI": (
            f"יצרנית איכותית שמייצרת מזומנים בעקביות ({fcf:.1f}%) עם ניהול פיננסי אחראי וממושמע שמחזיר ערך למשקיעים.",
            "💰 תזרים איכותי וניהול פיננסי אחראי",
            f"High-quality company consistently generating real cash ({fcf:.1f}%) with disciplined balance sheet management.",
            "Quality FCF & Disciplined Allocation",
        ),
        "ON": (
            f"שילוב מנצח של מומנטום עליות חזק בשנה האחרונה ({mom:+.1f}%) יחד עם יצירת מזומנים מוצקה ({fcf:.1f}%) מעל המתחרות.",
            "🚀 מומנטום עליות חזק ורווחיות מוצקה",
            f"Winning blend of powerful 12M momentum ({mom:+.1f}%) paired with superior real cash flow ({fcf:.1f}%) vs peers.",
            "Strong Momentum & Solid Cash Yield",
        ),
        "BAC": (
            f"מניית עוגן סולידית ויציבה שמספקת רשת ביטחון כשהשוק הכללי נכנס לפאניקה, עם תנודות רגועות של {vol:.1f}% בלבד.",
            "🛡️ עוגן סולידי ויציב להגנה על התיק",
            f"Solid defensive banking anchor providing downside safety when market turbulence hits, with low {vol:.1f}% vol.",
            "Defensive Banking Anchor",
        ),
        "MU": (
            f"מניה שנמצאת ברכבת הרים של עליות חזקות בשנה האחרונה ({mom:+.1f}%) ומספקת דחף צמיחה עוצמתי לתיק כולו.",
            "🚀 מומנטום אדיר ודחף צמיחה לתיק",
            f"Powerful high-momentum runner surging over the past year ({mom:+.1f}%), supplying pure growth velocity.",
            "High Momentum Growth Surge",
        ),
    }

    if sym in ticker_profiles:
        return ticker_profiles[sym]

    # Dynamic multi-factor rules for any non-preset constituent
    if zm >= 1.5:
        summary_he = "🚀 מומנטום אדיר ודחף צמיחה לתיק"
        summary_en = "High Momentum Growth Surge"
        rationale_he = f"מניה שנמצאת ברכבת הרים של עליות חזקות בשנה האחרונה ({mom:+.1f}%) ומספקת דחף צמיחה עוצמתי לתיק כולו."
        rationale_en = f"Powerful high-momentum runner surging over the past year ({mom:+.1f}%), supplying pure growth velocity."
    elif zv >= 1.9 and zs >= 0.7:
        summary_he = "💰 רווחיות מזומן אדירה ויציבות בהגנה"
        summary_en = "Top-Decile FCF & Defensive Stability"
        rationale_he = f"חברה שמייצרת כמויות אדירות של מזומן נקי ({fcf:.1f}%) ושומרת על תנודות מחיר רגועות יחסית ({vol:.1f}%), מה שמגן על התיק בנפילות."
        rationale_en = f"Produces massive free cash flow ({fcf:.1f}%) while keeping price swings calm ({vol:.1f}% vol), protecting the portfolio during pullbacks."
    elif zs >= 1.1 and fcf >= 3.5:
        summary_he = "🛡️ עוגן סולידי ויציב להגנה על התיק"
        summary_en = "Defensive Low-Vol Anchor"
        rationale_he = f"מניית עוגן סולידית ויציבה (תנודתיות {vol:.1f}% בלבד) שמספקת רשת ביטחון כשהשוק הכללי נכנס לפאניקה, לצד רווחיות בריאה ({fcf:.1f}% מזומן)."
        rationale_en = f"Solid and calm anchor stock (only {vol:.1f}% vol) acting as a safety net when the broader market panics, with steady cash yield ({fcf:.1f}%)."
    elif zs >= 1.1:
        summary_he = "🛡️ יציבות מקסימלית ועוגן הגנתי"
        summary_en = "Max Stability & Defensive Anchor"
        rationale_he = f"מניה עם תנודתיות נמוכה במיוחד ({vol:.1f}%) שמתפקדת כעוגן יציבות ושומרת על התיק מפני זעזועים, לצד מגמת צמיחה חיובית ({mom:+.1f}%)."
        rationale_en = f"Exceptionally low volatility ({vol:.1f}%) acting as a shock absorber against market swings, paired with positive momentum ({mom:+.1f}%)."
    elif mom >= 40.0 and zv >= 1.4:
        summary_he = "🚀 מומנטום עליות חזק ורווחיות מוצקה"
        summary_en = "Strong Momentum & Solid Cash Yield"
        rationale_he = f"שילוב מנצח של מומנטום עליות חזק בשנה האחרונה ({mom:+.1f}%) יחד עם יצירת מזומנים מוצקה ({fcf:.1f}%) מעל המתחרות."
        rationale_en = f"Winning blend of powerful 12M momentum ({mom:+.1f}%) paired with superior real cash flow ({fcf:.1f}%) vs peers."
    elif zv >= 1.8 and mom > 10.0:
        summary_he = "💰 יצירת מזומנים מעולה ומרווח ביטחון"
        summary_en = "High FCF Margin of Safety"
        rationale_he = f"חברה שמייצרת תזרים מזומנים חזק מאוד ({fcf:.1f}%) ונותנת מרווח ביטחון גבוה לצד מגמת מחיר חיובית בשוק ({mom:+.1f}%)."
        rationale_en = f"Exceptional cash flow generation ({fcf:.1f}%) providing a strong safety margin alongside a positive price trend ({mom:+.1f}%)."
    elif zv >= 1.8:
        summary_he = "💰 רווחיות מזומן מצוינת ומרווח ביטחון"
        summary_en = "High Cash Flow Generation"
        rationale_he = f"ענקית איכותית עם יצירת מזומנים מצוינת ({fcf:.1f}%) המעניקה מרווח ביטחון גבוה ועמידות עסקית."
        rationale_en = f"High-tier free cash flow yield ({fcf:.1f}%) providing a superior margin of safety and business resilience."
    elif zv >= 1.5:
        summary_he = "💰 תזרים איכותי וניהול פיננסי אחראי"
        summary_en = "Quality FCF & Disciplined Allocation"
        rationale_he = f"יצרנית איכותית שמייצרת מזומנים בעקביות ({fcf:.1f}%) עם ניהול פיננסי אחראי וממושמע שמחזיר ערך למשקיעים."
        rationale_en = f"High-quality company consistently generating real cash ({fcf:.1f}%) with disciplined balance sheet management."
    else:
        summary_he = "⚖️ פרופיל רב-גורמי מאוזן"
        summary_en = "Balanced Multi-Factor"
        rationale_he = f"פרופיל מאוזן המשלב יצירת מזומנים בריאה ({fcf:.1f}%) יחד עם תנודתיות מבוקרת ({vol:.1f}%) לשמירה על יציבות התיק."
        rationale_en = f"Balanced profile combining healthy cash generation ({fcf:.1f}%) with controlled volatility ({vol:.1f}% vol)."

    return rationale_he, summary_he, rationale_en, summary_en


def get_current_signals(force_refresh: bool = False, top_n: int = TOP_N) -> dict:
    """
    Run the full signal pipeline using data up to today.

    Returns a dict:
        {
            'as_of_date'    : pd.Timestamp,
            'regime'        : 'Risk-ON' | 'Risk-OFF',
            'risk_on'       : bool,
            'spy_price'     : float,
            'spy_sma'       : float,
            'macro_regime'  : dict,
            'portfolio'     : pd.DataFrame,   # top-N with scores + weights + rationales
            'full_scores'   : pd.DataFrame,   # all qualified tickers, ranked
            'fundamentals'  : pd.DataFrame,
        }
    """
    today_str = date.today().strftime("%Y-%m-%d")

    # ── 1. Download data ──────────────────────────────────────────────────
    print(f"\n[1/4] Fetching price data up to {today_str} ...")
    all_tickers = list(set(UNIVERSE_TICKERS + [MARKET_TICKER, BOND_TICKER, CASH_TICKER]))
    prices = pd.DataFrame()
    try:
        prices = fetch_price_data(
            all_tickers,
            start=DATA_START_DATE,
            end=today_str,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logger.warning("Price download encountered exception: %s", exc)

    if prices is None or prices.empty:
        logger.info("Direct price fetch empty; loading latest comprehensive local cache...")
        pkl_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "prices_*.pkl"))
        for pkl in sorted(pkl_files, key=os.path.getmtime, reverse=True):
            try:
                with open(pkl, "rb") as f:
                    c_df = pickle.load(f)
                    if isinstance(c_df, pd.DataFrame) and MARKET_TICKER in c_df.columns:
                        prices = c_df
                        break
            except Exception:
                continue

    prices = prices.dropna(how="all").ffill()
    print(f"      {prices.shape[1]} tickers  |  {prices.index[-1].date()} last price")

    # ── 2. Fundamentals ───────────────────────────────────────────────────
    print("[2/4] Fetching fundamentals ...")
    fundamentals = fetch_fundamentals(UNIVERSE_TICKERS, force_refresh=force_refresh)

    # ── 3. Macro regime ───────────────────────────────────────────────────
    print("[3/4] Evaluating macro regime ...")
    spy_prices    = prices[MARKET_TICKER].dropna()
    spy_monthly   = spy_prices.resample("ME").last()
    macro_signal  = compute_macro_signal(spy_monthly, sma_months=MACRO_SMA_MONTHS)

    # Use the most recent complete month-end available
    as_of_date    = spy_monthly.index[-1]
    spy_price     = float(spy_monthly.iloc[-1])
    spy_sma       = float(spy_monthly.iloc[-MACRO_SMA_MONTHS:].mean())
    sma_diff      = ((spy_price - spy_sma) / spy_sma) * 100.0

    risk_on       = bool(macro_signal.iloc[-1])
    regime        = "Risk-ON  (100% Equities)" if risk_on else "Risk-OFF (50% Bonds + 50% Equities)"

    macro_regime_info = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "spy_price": round(spy_price, 2),
        "spy_10m_sma": round(spy_sma, 2),
        "distance_pct": round(sma_diff, 2),
        "status": "Risk-ON" if risk_on else "Risk-OFF",
        "regime_label": "Risk-ON (100% Equities)" if risk_on else "Risk-OFF (50% Bonds + 50% Equities)",
        "target_allocation": "100% Equities" if risk_on else "50% Bonds + 50% Equities",
        "equity_allocation_pct": 100.0 if risk_on else 50.0,
        "bond_allocation_pct": 0.0 if risk_on else 50.0,
        "buffer_text": f"{sma_diff:+.2f}% {'above' if sma_diff >= 0 else 'below'} SMA safety threshold",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 4. Factor scoring ─────────────────────────────────────────────────
    print("[4/4] Scoring universe ...")
    equity_universe = [
        t for t in UNIVERSE_TICKERS
        if t in prices.columns and t not in (MARKET_TICKER, BOND_TICKER, CASH_TICKER)
    ]
    eval_date = pd.to_datetime(prices.index[-1])
    qualified = apply_quality_filter(equity_universe, fundamentals, as_of_date=eval_date)

    factor_df = compute_factor_details(prices[qualified], qualified, as_of_date, fundamentals=fundamentals)

    if factor_df.empty:
        print("ERROR: No tickers scored — check price data availability.")
        sys.exit(1)

    # Merge fundamentals into factor table
    fund_sub = fundamentals[["fcf", "de_ratio"]].copy()
    fund_sub.index.name = "ticker"
    full_scores = factor_df.join(fund_sub, how="left")

    # Top-N selection with generalized deduplication (one ticker per underlying company)
    selected_tickers = select_portfolio(full_scores["combined_score"], top_n=top_n, fundamentals=fundamentals)
    top_tickers = full_scores.loc[selected_tickers]

    # Compute weights
    if risk_on:
        eq_weight   = 1.0 / top_n
        bond_weight = 0.0
    else:
        eq_weight   = DEFENSIVE_EQUITY_ALLOC / top_n
        bond_weight = DEFENSIVE_BOND_ALLOC

    top_tickers = top_tickers.copy()
    top_tickers["target_weight"] = eq_weight
    top_tickers["rank"] = range(1, len(top_tickers) + 1)

    # Generate 1-sentence factor rationales and formatted metrics
    rationales = []
    rationales_he = []
    rationales_en = []
    summaries = []
    summaries_en = []
    fcf_yields = []
    vols = []
    moms = []

    for _, row in top_tickers.iterrows():
        r_dict = row.to_dict()
        r_dict["ticker"] = row.name
        rat_he, smy_he, rat_en, smy_en = generate_factor_rationale(r_dict)
        rationales.append(rat_he)
        rationales_he.append(rat_he)
        rationales_en.append(rat_en)
        summaries.append(smy_he)
        summaries_en.append(smy_en)
        fcf_yields.append(round(float(row.get("value_raw", 0.0) or 0.0) * 100.0, 2))
        vols.append(round(float(row.get("stability_raw", 0.0) or 0.0) * 100.0, 2))
        moms.append(round(float(row.get("momentum_raw", 0.0) or 0.0) * 100.0, 2))

    top_tickers["fcf_yield_pct"] = fcf_yields
    top_tickers["volatility_90d_pct"] = vols
    top_tickers["momentum_12m_1m_pct"] = moms
    top_tickers["rationale"] = rationales
    top_tickers["rationale_he"] = rationales_he
    top_tickers["rationale_en"] = rationales_en
    top_tickers["rationale_summary"] = summaries
    top_tickers["rationale_summary_en"] = summaries_en

    return {
        "as_of_date":   as_of_date,
        "regime":       regime,
        "risk_on":      risk_on,
        "spy_price":    spy_price,
        "spy_sma":      spy_sma,
        "macro_regime": macro_regime_info,
        "bond_weight":  bond_weight,
        "bond_ticker":  BOND_TICKER,
        "eq_weight":    eq_weight,
        "portfolio":    top_tickers,
        "full_scores":  full_scores,
        "fundamentals": fundamentals,
    }


# ─────────────────────────────────────────────────────────────────────────────

def print_signals(result: dict, top_n: int = TOP_N) -> None:
    """Print a richly formatted signal table to stdout."""
    port       = result["portfolio"]
    as_of      = result["as_of_date"].strftime("%Y-%m-%d")
    spy_price  = result["spy_price"]
    spy_sma    = result["spy_sma"]
    sma_diff   = (spy_price / spy_sma - 1) * 100
    risk_on    = result["risk_on"]

    W = 98
    print()
    print("=" * W)
    print("  THE VALUE HUNTER — CURRENT PORTFOLIO SIGNALS (30% MOM / 30% STAB / 40% VALUE)")
    print("=" * W)
    print(f"  Signal date : {as_of}")
    print(f"  SPY price   : ${spy_price:,.2f}   |   10M SMA: ${spy_sma:,.2f}   |   Delta: {sma_diff:+.2f}%")
    print(f"  Macro regime: {'[RISK-ON ]' if risk_on else '[RISK-OFF]'}  {result['regime']}")
    if not risk_on:
        print(f"  Bond leg    : {result['bond_weight']*100:.0f}% {result['bond_ticker']}  "
              f"(held alongside the equity positions)")
    print("-" * W)

    # Column widths
    COL = {
        "rank":           4,
        "ticker":         8,
        "z_momentum":     9,
        "z_stability":    9,
        "z_value":        9,
        "combined_score":10,
        "momentum_raw":  11,
        "stability_raw":  9,
        "value_raw":      9,
        "fcf_b":          9,
        "de_ratio":       7,
        "weight":         7,
    }

    header = (
        f"{'#':>{COL['rank']}} "
        f"{'Ticker':<{COL['ticker']}} "
        f"{'Z-Mom':>{COL['z_momentum']}} "
        f"{'Z-Stab':>{COL['z_stability']}} "
        f"{'Z-Value':>{COL['z_value']}} "
        f"{'Combined':>{COL['combined_score']}} "
        f"{'12M-1M Ret':>{COL['momentum_raw']}} "
        f"{'90D Vol':>{COL['stability_raw']}} "
        f"{'FCF Yld':>{COL['value_raw']}} "
        f"{'FCF ($B)':>{COL['fcf_b']}} "
        f"{'D/E':>{COL['de_ratio']}} "
        f"{'Weight':>{COL['weight']}}"
    )
    print(header)
    print("-" * W)

    for _, row in port.iterrows():
        ticker  = row.name
        fcf_b   = f"{row['fcf']/1e9:.2f}" if pd.notna(row.get("fcf")) and row["fcf"] else "N/A"
        de      = f"{row['de_ratio']:.2f}" if pd.notna(row.get("de_ratio")) else "N/A"
        z_val   = row.get("z_value", 0.0)
        val_raw = row.get("value_raw", np.nan)

        line = (
            f"{int(row['rank']):>{COL['rank']}} "
            f"{ticker:<{COL['ticker']}} "
            f"{row['z_momentum']:>{COL['z_momentum']}.3f} "
            f"{row['z_stability']:>{COL['z_stability']}.3f} "
            f"{z_val:>{COL['z_value']}.3f} "
            f"{row['combined_score']:>{COL['combined_score']}.3f} "
            f"{_fmt_pct(row['momentum_raw']):>{COL['momentum_raw']}} "
            f"{_fmt_pct(row['stability_raw']):>{COL['stability_raw']}} "
            f"{_fmt_pct(val_raw):>{COL['value_raw']}} "
            f"{fcf_b:>{COL['fcf_b']}} "
            f"{de:>{COL['de_ratio']}} "
            f"{_fmt_pct(row['target_weight']):>{COL['weight']}}"
        )
        print(line)

    print("-" * W)
    total_eq = result["eq_weight"] * len(port)
    total_bnd = result["bond_weight"]
    total_cash = max(0.0, 1.0 - total_eq - total_bnd)
    print(f"  Equity allocation : {total_eq*100:.1f}%   "
          f"Bond ({result['bond_ticker']}): {total_bnd*100:.1f}%   "
          f"Cash: {total_cash*100:.1f}%")
    print(f"  Qualified universe: {len(result['full_scores'])} tickers "
          f"(after FCF + D/E filter)")
    print("=" * W)

    print("\n  \"WHY THIS STOCK?\" MULTI-FACTOR MODEL RATIONALES:")
    print("-" * W)
    for _, row in port.iterrows():
        fcf_y = row.get("fcf_yield_pct", row.get("value_raw", 0.0) * 100.0)
        v90 = row.get("volatility_90d_pct", row.get("stability_raw", 0.0) * 100.0)
        m12 = row.get("momentum_12m_1m_pct", row.get("momentum_raw", 0.0) * 100.0)
        summary = row.get("rationale_summary", "Top Multi-Factor")
        rat = row.get("rationale", "")
        print(f"  #{int(row['rank']):<2} {row.name:<5} | [{summary}] [FCF: {fcf_y:.1f}% | 90D Vol: {v90:.1f}% | 12M Mom: {m12:+.1f}%]")
        print(f"      -> \"{rat}\"")
    print("=" * W)


def export_current_csv(result: dict, out_path: str) -> None:
    """Export the current portfolio signals and macro regime to CSV and JSON."""
    port = result["portfolio"].copy()

    # Clean up for export
    export_cols = [
        "rank", "combined_score",
        "fcf_yield_pct", "volatility_90d_pct", "momentum_12m_1m_pct",
        "z_value", "z_stability", "z_momentum",
        "rationale_summary", "rationale_summary_en",
        "rationale", "rationale_he", "rationale_en",
        "value_raw", "stability_raw", "momentum_raw",
        "fcf", "de_ratio", "target_weight",
    ]
    # Ensure all columns exist
    for c in export_cols:
        if c not in port.columns:
            port[c] = np.nan

    export = port[export_cols].copy()
    export.index.name = "ticker"
    export.insert(0, "signal_date", result["as_of_date"].strftime("%Y-%m-%d"))
    export.insert(1, "regime",      "Risk-ON" if result["risk_on"] else "Risk-OFF")
    export.insert(2, "bond_ticker", result["bond_ticker"] if not result["risk_on"] else "")
    export.insert(3, "bond_weight", result["bond_weight"])

    # Round numeric columns
    for col in ["z_momentum", "z_stability", "z_value", "combined_score"]:
        export[col] = export[col].round(4)
    export["momentum_raw"]     = export["momentum_raw"].round(4)
    export["stability_raw"]    = export["stability_raw"].round(6)
    export["value_raw"]        = export["value_raw"].round(6)
    export["target_weight"]    = export["target_weight"].round(4)
    export["fcf_yield_pct"]    = export["fcf_yield_pct"].round(2)
    export["volatility_90d_pct"] = export["volatility_90d_pct"].round(2)
    export["momentum_12m_1m_pct"] = export["momentum_12m_1m_pct"].round(2)

    out_full = os.path.join(os.path.dirname(__file__), out_path)
    export.to_csv(out_full)
    print(f"\n  [CSV] Current signals saved -> {out_full}")

    # Export structured JSON companion files for the API server & frontend
    signals_json_path = os.path.join(os.path.dirname(__file__), "current_portfolio_signals.json")
    macro_json_path = os.path.join(os.path.dirname(__file__), "current_macro_regime.json")

    records = export.reset_index().to_dict(orient="records")
    payload = {
        "as_of_date": result["as_of_date"].strftime("%Y-%m-%d"),
        "macro_regime": result["macro_regime"],
        "count": len(records),
        "signals": records,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(signals_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [JSON] Signals JSON saved -> {signals_json_path}")

    with open(macro_json_path, "w", encoding="utf-8") as f:
        json.dump(result["macro_regime"], f, indent=2)
    print(f"  [JSON] Macro regime JSON saved -> {macro_json_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    result = get_current_signals(force_refresh=args.refresh, top_n=args.top)
    print_signals(result, top_n=args.top)
    export_current_csv(result, out_path=args.out)

