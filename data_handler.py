"""
data_handler.py — Downloads and caches price + fundamental data.

All prices returned are ADJUSTED CLOSE prices to account for dividends
and stock splits, preventing look-ahead bias in the price series.
"""

import os
import pickle
import hashlib
import logging
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config import CACHE_DIR, DATA_START_DATE, START_DATE, END_DATE

logger = logging.getLogger(__name__)

# Resolve cache dir relative to this file (not the CWD)
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE_DIR, CACHE_DIR)
os.makedirs(_CACHE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> str:
    """Return an OS-safe cache file path for the given key (hash to avoid long paths)."""
    # Use a short MD5 hash to avoid Windows 260-char path limit
    key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
    # Keep a human-readable prefix
    prefix = key.split("_")[0][:20].replace("/", "_").replace(":", "_")
    return os.path.join(_CACHE_DIR, f"{prefix}_{key_hash}.pkl")


def _load_cache(key: str):
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(key: str, obj) -> None:
    with open(_cache_path(key), "wb") as f:
        pickle.dump(obj, f)

# ── Price data ────────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: List[str],
    start: str = DATA_START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices.
    Shape: (dates, tickers).
    """
    cache_key = f"prices_{start}_{end}_{'_'.join(sorted(tickers))}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info("Loaded price data from cache.")
            return cached

    logger.info("Downloading price data for %d tickers …", len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # returns adjusted OHLCV
        progress=False,
        threads=True,
    )

    # yfinance 1.4+ always returns MultiIndex columns (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    # Drop tickers with insufficient history (< 252 days)
    # Drop tickers with insufficient history (< 252 days) – disabled to retain all tickers for long backtest
    # close = close.dropna(axis=1, thresh=252)
    # Keep all tickers regardless of missing data; factor calculations will handle NaNs
    
    close = close.sort_index()

    if not close.empty:
        _save_cache(cache_key, close)
        logger.info("Downloaded and cached %d tickers, %d rows.", close.shape[1], close.shape[0])
    else:
        logger.warning("Downloaded price data is empty; not caching.")
    return close


# ── Fundamental data (TTM FCF & Debt-to-Equity) ──────────────────────────────

def fetch_fundamentals(
    tickers: List[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch TTM Free Cash Flow, Debt-to-Equity ratio, and record the fetch date for each ticker.
    Returns a DataFrame indexed by ticker with columns ['fcf', 'de_ratio', 'company_name', 'cik', 'website', 'fetch_date'].

    The `fetch_date` is the UTC date when the data was downloaded. In the backtest we enforce a
    conservative 45‑day reporting lag, ensuring only fundamentals older than `as_of_date - 45d`
    are used.
    """
    cache_key = f"fundamentals_{'_'.join(sorted(tickers))}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info("Loaded fundamentals from cache.")
            return cached

    logger.info("Fetching fundamentals for %d tickers …", len(tickers))
    records = []
    download_date = pd.Timestamp.utcnow().date()
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            # TTM Free Cash Flow
            fcf = info.get("freeCashflow", None)

            # Debt-to-Equity (reported as a ratio, e.g. 1.25 means 125%)
            de = info.get("debtToEquity", None)
            if de is not None:
                de = de / 100.0  # normalize to ratio form

            company_name = info.get("longName") or info.get("shortName")
            cik = info.get("cik")
            website = info.get("website")

            records.append({
                "ticker": ticker,
                "fcf": fcf,
                "de_ratio": de,
                "company_name": company_name,
                "cik": cik,
                "website": website,
                "fetch_date": download_date,
            })
        except Exception as exc:
            logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
            records.append({
                "ticker": ticker,
                "fcf": None,
                "de_ratio": None,
                "company_name": None,
                "cik": None,
                "website": None,
                "fetch_date": download_date,
            })

    # Build DataFrame, cache, and return fundamentals
    fundamentals_df = pd.DataFrame(records).set_index('ticker')
    _save_cache(cache_key, fundamentals_df)
    return fundamentals_df

# ── Open prices (for execution simulation) ───────────────────────────────────

def fetch_open_prices(
    tickers: List[str],
    start: str = DATA_START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return daily OPEN prices (adjusted) for execution simulation.
    Signals are generated on the last trading day of the month;
    trades are executed at the OPEN of the 1st trading day of the next month.
    """
    cache_key = f"open_{start}_{end}_{'_'.join(sorted(tickers))}"
    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    logger.info("Downloading open prices …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        opens = raw["Open"].copy()
        if isinstance(opens, pd.Series):
            opens = opens.to_frame(name=tickers[0])
    else:
        opens = raw[["Open"]].copy()
        opens.columns = tickers[:1]

    opens = opens.sort_index()
    if not opens.empty:
        _save_cache(cache_key, opens)
        logger.info("Downloaded and cached %d tickers, %d rows of open prices.", opens.shape[1], opens.shape[0])
    else:
        logger.warning("Downloaded open prices are empty; not caching.")
    return opens
