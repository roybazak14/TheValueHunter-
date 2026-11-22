"""
strategy.py — Signal generation for The Value Hunter.

ALL signals are generated using data available up to (and including) the
LAST TRADING DAY of each month.  Execution happens at the OPEN of the
FIRST TRADING DAY of the FOLLOWING month.  This enforces strict
no-look-ahead-bias discipline.
"""

import os
import re
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Set, Union

import numpy as np
import pandas as pd

from config import (
    MARKET_TICKER,
    MACRO_SMA_MONTHS,
    BOND_TICKER,
    CASH_TICKER,
    DEFENSIVE_BOND_ALLOC,
    DEFENSIVE_EQUITY_ALLOC,
    FCF_MIN,
    DE_BOTTOM_PCT,
    MOMENTUM_MIN,
    MAX_FCF_YIELD,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER,
    ZSCORE_CLIP,
    MOMENTUM_MONTHS,
    MOMENTUM_SKIP,
    STABILITY_DAYS,
    MOMENTUM_WEIGHT,
    STABILITY_WEIGHT,
    VALUE_WEIGHT,
    TOP_N,
    EQUAL_WEIGHT,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


# ── Macro Overlay ─────────────────────────────────────────────────────────────

def compute_macro_signal(
    spy_monthly_close: pd.Series,
    sma_months: int = MACRO_SMA_MONTHS,
) -> pd.Series:
    """
    Compute the 10-month SMA of SPY monthly close prices.

    Returns a boolean Series:
        True  → risk-on  (SPY > SMA → 100% equities)
        False → risk-off (SPY < SMA → 50% bonds + 50% equities)

    The signal on date T uses the monthly close on date T (i.e. the
    last trading day of that month).  It is applied at the open of
    month T+1 — no look-ahead.
    """
    sma = spy_monthly_close.rolling(window=sma_months, min_periods=sma_months).mean()
    risk_on = spy_monthly_close > sma
    return risk_on  # indexed by month-end dates


# ── Quality Filter ────────────────────────────────────────────────────────────

def apply_quality_filter(
    universe: List[str],
    fundamentals: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
    fcf_min: float = FCF_MIN,
    de_bottom_pct: float = DE_BOTTOM_PCT,
) -> List[str]:
    """
    Apply quality screens to the universe, respecting a point‑in‑time reporting lag.

    Parameters
    ----------
    universe: list of tickers to screen.
    fundamentals: DataFrame with columns ['fcf', 'de_ratio', 'fetch_date'].
    as_of_date: the rebalance date (month‑end) for which the filter is evaluated.
    fcf_min, de_bottom_pct: thresholds as before.

    The function discards any ticker whose `fetch_date` is newer than
    `as_of_date - pd.Timedelta(days=45)`, implementing a conservative
    45‑day reporting lag.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now()
    else:
        as_of_date = pd.to_datetime(as_of_date)

    df = fundamentals.copy()
    df = df[df.index.isin(universe)]

    # Preserve original fundamentals before applying reporting‑lag filter
    df_original = df.copy()
    # Apply reporting‑lag filter; keep rows where fetch_date is missing (unknown) or older than lag cutoff.
    if "fetch_date" in df.columns:
        df["fetch_date"] = pd.to_datetime(df["fetch_date"], errors="coerce")
        lag_cutoff = as_of_date - pd.Timedelta(days=45)
        df_lagged = df[(df["fetch_date"].isna()) | (df["fetch_date"] <= lag_cutoff)]
        if df_lagged.empty:
            logger.debug("Reporting‑lag filter removed all rows; skipping lag filter.")
            df = df_original
        else:
            df = df_lagged
    else:
        logger.debug("fetch_date column missing; skipping reporting‑lag filter.")





    # --- FCF filter: keep positive FCF (allow missing values) ---
    if "fcf" in df.columns:
        df = df[(df["fcf"].isna()) | (df["fcf"] > fcf_min)]
    else:
        logger.debug("FCF column missing in fundamentals; skipping FCF filter.")

    # --- D/E filter: exclude bottom 20% (highest D/E), keep missing values ---
    if "de_ratio" in df.columns:
        de_valid = df["de_ratio"].dropna()
        if len(de_valid) > 0:
            threshold = de_valid.quantile(1.0 - de_bottom_pct)  # 80th percentile
            df = df[(df["de_ratio"].isna()) | (df["de_ratio"] <= threshold)]
    else:
        logger.debug("D/E column missing in fundamentals; skipping D/E filter.")
    # After filters, prepare list of qualified tickers
    qualified = df.index.tolist()
    if not isinstance(qualified, list):
        qualified = list(qualified)
    # If not enough tickers survive the quality screens, fall back to the full universe
    if len(qualified) < TOP_N:
        logger.warning(
            "Quality filter left %d tickers (< TOP_N=%d) for %s; using full universe.",
            len(qualified), TOP_N, as_of_date.date()
        )
        qualified = list(universe)

    logger.debug("Quality filter: %d → %d tickers.", len(universe), len(qualified))
    return qualified





# ── Factor Signals ────────────────────────────────────────────────────────────

def compute_momentum(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    months_back: int = MOMENTUM_MONTHS,
    skip_months: int = MOMENTUM_SKIP,
) -> pd.Series:
    """
    12M-1M momentum: return from T-12M to T-1M.

    Using only data up to `as_of_date` (last day of the current month).
    """
    # Ensure we only use data up to the rebalance date (PIT)
    prices = prices.loc[:as_of_date]
    # End of momentum window = 1 month ago (skip last month)
    end_date = as_of_date - pd.DateOffset(months=skip_months)
    # Start of window = 12 months before the end
    start_date = end_date - pd.DateOffset(months=months_back)

    # Get the closest available trading days
    available = prices.index[prices.index <= as_of_date]
    if len(available) == 0:
        return pd.Series(dtype=float)

    # Price at end_date (1 month ago)
    end_idx = prices.index.get_indexer([end_date], method="ffill")[0]
    if end_idx < 0:
        return pd.Series(dtype=float)

    # Price at start_date (13 months ago)
    start_idx = prices.index.get_indexer([start_date], method="ffill")[0]
    if start_idx < 0 or start_idx >= end_idx:
        return pd.Series(dtype=float)

    p_end   = prices.iloc[end_idx]
    p_start = prices.iloc[start_idx]

    momentum = (p_end / p_start) - 1.0
    return momentum


def compute_stability(
    prices: Union[pd.DataFrame, pd.Series],
    as_of_date: pd.Timestamp,
    window_days: int = STABILITY_DAYS,
) -> Union[pd.Series, float]:
    """
    90-day rolling daily return standard deviation (annualized: std * sqrt(252)).
    Lower is better — stocks with lower volatility score higher.

    Uses only data up to `as_of_date`.
    """
    # Ensure we only use data up to the rebalance date (PIT)
    prices = prices.loc[:as_of_date]
    hist = prices[prices.index <= as_of_date].tail(window_days + 1)

    if len(hist) < 20:
        return pd.Series(dtype=float) if isinstance(prices, pd.DataFrame) else np.nan

    # Calculate daily returns (skip first NaN row)
    daily_ret = hist.pct_change().iloc[1:]

    if isinstance(hist, pd.DataFrame):
        # Column-wise standard deviation (skips NaNs per ticker, avoiding dropna() row purge)
        std = daily_ret.std(skipna=True)
        # Require at least 20 valid trading days for each ticker in the window
        valid_counts = daily_ret.count()
        std[valid_counts < 20] = np.nan
        return std * np.sqrt(252)
    else:
        if daily_ret.count() < 20:
            return np.nan
        return float(daily_ret.std(skipna=True) * np.sqrt(252))


_SHARES_CACHE: Optional[Dict[str, float]] = None
_FUNDAMENTALS_CACHE: Optional[pd.DataFrame] = None


def _get_shares_dict() -> Dict[str, float]:
    """Load and cache shares outstanding dictionary from local cache."""
    global _SHARES_CACHE
    if _SHARES_CACHE is not None:
        return _SHARES_CACHE
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CACHE_DIR, "shares_cache.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                _SHARES_CACHE = pickle.load(f)
                return _SHARES_CACHE
        except Exception as e:
            logger.warning("Failed to load shares_cache.pkl: %s", e)
    _SHARES_CACHE = {}
    return _SHARES_CACHE


def _get_default_fundamentals() -> pd.DataFrame:
    """Fallback loader for fundamentals if not explicitly passed."""
    global _FUNDAMENTALS_CACHE
    if _FUNDAMENTALS_CACHE is not None:
        return _FUNDAMENTALS_CACHE
    try:
        from data_handler import fetch_fundamentals
        from config import UNIVERSE_TICKERS
        _FUNDAMENTALS_CACHE = fetch_fundamentals(UNIVERSE_TICKERS, force_refresh=False)
        return _FUNDAMENTALS_CACHE
    except Exception:
        return pd.DataFrame()


def compute_value(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    fundamentals: Optional[pd.DataFrame] = None,
    shares_dict: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Compute Point-in-Time Free Cash Flow Yield (Value factor):
        FCF Yield = FCF / Market Cap = FCF / (Price_t * Shares)

    Uses only price data available on or before `as_of_date` (strictly PIT).
    """
    if fundamentals is None or fundamentals.empty:
        fundamentals = _get_default_fundamentals()
    if shares_dict is None:
        shares_dict = _get_shares_dict()

    prices_pit = prices.loc[:as_of_date]
    if prices_pit.empty:
        return pd.Series(dtype=float)

    p_slice = prices_pit.iloc[-1]
    fcf_series = fundamentals["fcf"] if (not fundamentals.empty and "fcf" in fundamentals.columns) else pd.Series(dtype=float)

    val_yield = {}
    for t in prices.columns:
        p = p_slice.get(t)
        sh = shares_dict.get(t)
        f = fcf_series.get(t) if t in fcf_series.index else None

        if p and not pd.isna(p) and p > 0 and sh and sh > 0 and f and not pd.isna(f) and f > 0:
            mkt_cap = p * sh
            raw_yield = float(f) / float(mkt_cap)
            val_yield[t] = min(raw_yield, MAX_FCF_YIELD)
        else:
            val_yield[t] = 0.0

    return pd.Series(val_yield)


def compute_combined_score(
    prices: pd.DataFrame,
    qualified_tickers: List[str],
    as_of_date: pd.Timestamp,
    momentum_weight: float = MOMENTUM_WEIGHT,
    stability_weight: float = STABILITY_WEIGHT,
    value_weight: float = VALUE_WEIGHT,
    fundamentals: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Cross-sectional Z-score of momentum, stability, and value (FCF Yield), combined.

    Combined Score = (0.30 * Z-Mom) + (0.30 * Z-Stab) + (0.40 * Z-Value)
    Note: stability is inverted so higher score = more stable.
    Returns a Series sorted descending by combined score.
    """
    df = compute_factor_details(
        prices, qualified_tickers, as_of_date,
        momentum_weight, stability_weight, value_weight,
        fundamentals=fundamentals,
    )
    if df.empty:
        return pd.Series(dtype=float)
    return df["combined_score"]


def compute_factor_details(
    prices: pd.DataFrame,
    qualified_tickers: List[str],
    as_of_date: pd.Timestamp,
    momentum_weight: float = MOMENTUM_WEIGHT,
    stability_weight: float = STABILITY_WEIGHT,
    value_weight: float = VALUE_WEIGHT,
    fundamentals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full 3-factor breakdown per ticker — used for reporting and signal export.

    Includes:
      1. Value-Trap Momentum Floor (`MOMENTUM_MIN = 0.0`): strictly excludes
         falling-knife tickers with negative 12M-1M price return.
      2. FCF Yield Cap (`MAX_FCF_YIELD = 0.20`): caps abnormal/distressed FCF yields.
      3. Winsorization (5th/95th percentiles) + Z-score clipping (`[-2.5, +2.5]`)
         to prevent single-stock outliers from distorting cross-sectional scoring.

    Returns a DataFrame (sorted descending by combined_score) with columns:
        momentum_raw   : 12M-1M raw return
        stability_raw  : 90-day annualized return std dev (lower = more stable)
        value_raw      : Free Cash Flow Yield (FCF / Market Cap, capped at MAX_FCF_YIELD)
        z_momentum     : winsorized & clipped cross-sectional Z-score of momentum
        z_stability    : winsorized & clipped cross-sectional Z-score of inverted stability
        z_value        : winsorized & clipped cross-sectional Z-score of FCF Yield
        combined_score : weighted combination (0.30*Z_Mom + 0.30*Z_Stab + 0.40*Z_Value)
    """
    sub_prices = prices[[t for t in qualified_tickers if t in prices.columns]]

    mom  = compute_momentum(sub_prices, as_of_date)
    stab = compute_stability(sub_prices, as_of_date)
    val  = compute_value(sub_prices, as_of_date, fundamentals=fundamentals)

    common = mom.dropna().index.intersection(stab.dropna().index).intersection(val.index)
    mom    = mom[common]
    stab   = stab[common]
    val    = val[common].clip(upper=MAX_FCF_YIELD)

    # Value-Trap Momentum Floor: strictly exclude tickers with negative 12M-1M price return
    positive_mom_idx = mom[mom > MOMENTUM_MIN].index
    if len(positive_mom_idx) >= TOP_N:
        mom  = mom[positive_mom_idx]
        stab = stab[positive_mom_idx]
        val  = val[positive_mom_idx]
    else:
        logger.warning(
            "Only %d tickers have momentum > %.2f on %s (< TOP_N=%d); retaining top momentum names.",
            len(positive_mom_idx), MOMENTUM_MIN, as_of_date, TOP_N
        )

    if len(mom) < 5:
        logger.warning("Too few tickers (%d) for factor scoring on %s", len(mom), as_of_date)
        return pd.DataFrame()

    def zscore(s: pd.Series) -> pd.Series:
        # 1. Winsorize raw factor series at 5th and 95th percentiles
        low_q = s.quantile(WINSORIZE_LOWER)
        high_q = s.quantile(WINSORIZE_UPPER)
        s_win = s.clip(lower=low_q, upper=high_q)

        # 2. Standardize and cap Z-scores to [-ZSCORE_CLIP, +ZSCORE_CLIP]
        mu, sigma = s_win.mean(), s_win.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index)
        z = (s_win - mu) / sigma
        return z.clip(lower=-ZSCORE_CLIP, upper=ZSCORE_CLIP)

    z_mom  = zscore(mom)
    z_stab = zscore(-stab)   # invert: lower std → higher score
    z_val  = zscore(val)     # higher FCF yield → higher score

    combined = (momentum_weight * z_mom) + (stability_weight * z_stab) + (value_weight * z_val)

    df = pd.DataFrame({
        "momentum_raw":   mom,
        "stability_raw":  stab,
        "value_raw":      val,
        "z_momentum":     z_mom,
        "z_stability":    z_stab,
        "z_value":        z_val,
        "combined_score": combined,
    })
    return df.sort_values("combined_score", ascending=False)


# ── Company Identifier / Multi-Share Class Deduplication ─────────────────────

def normalize_company_name(name: Optional[str]) -> str:
    """Normalize company name to identify duplicate share classes of the same company."""
    if not name or pd.isna(name):
        return ""
    name_str = str(name).lower()
    # Remove class markers: Class A, Class B, Class C, Cl A, Cl C, etc.
    name_str = re.sub(r"\bclass\s+[a-z0-9]\b", "", name_str)
    name_str = re.sub(r"\bcl\s+[a-z0-9]\b", "", name_str)
    name_str = re.sub(r"\bseries\s+[a-z0-9]\b", "", name_str)
    # Remove legal entities and common suffixes
    name_str = re.sub(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|nv|sa|holding|holdings)\b", "", name_str)
    # Strip non-alphanumeric characters
    name_str = re.sub(r"[^a-z0-9]", "", name_str)
    return name_str.strip()


def get_company_key(ticker: str, fundamentals: Optional[pd.DataFrame] = None) -> str:
    """
    Determine a unique canonical company key for a ticker.
    Uses:
      1. CIK or normalized company name from fundamentals if present.
      2. Website domain / identifier if present.
      3. Robust standard dual-class pattern matching (e.g. GOOG/GOOGL, FOX/FOXA, NWSA/NWS, UA/UAA, BRK.A/BRK.B).
    """
    ticker_clean = ticker.upper().replace(".", "-").strip()

    # 1. Check fundamentals metadata
    if fundamentals is not None and ticker in fundamentals.index:
        row = fundamentals.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # CIK identifier is unique to the corporate entity
        cik = row.get("cik")
        if pd.notna(cik) and str(cik).strip():
            return f"CIK_{str(cik).strip()}"

        # Normalized company name
        c_name = row.get("company_name")
        norm_name = normalize_company_name(c_name)
        if norm_name and len(norm_name) >= 3:
            return f"NAME_{norm_name}"

        # Website domain
        website = row.get("website")
        if pd.notna(website) and str(website).strip():
            domain = re.sub(r"^https?://(www\.)?", "", str(website).lower()).split("/")[0].strip()
            if domain:
                return f"WEB_{domain}"

    # 2. Known dual-class clusters
    DUAL_CLASS_MAP = {
        "GOOG": "ALPHABET",
        "GOOGL": "ALPHABET",
        "FOX": "FOXCORP",
        "FOXA": "FOXCORP",
        "NWS": "NEWSCORP",
        "NWSA": "NEWSCORP",
        "UA": "UNDERARMOUR",
        "UAA": "UNDERARMOUR",
        "BRK-A": "BERKSHIRE",
        "BRK-B": "BERKSHIRE",
        "BRK.A": "BERKSHIRE",
        "BRK.B": "BERKSHIRE",
        "CWEN": "CLEARWAY",
        "CWEN-A": "CLEARWAY",
        "CWEN.A": "CLEARWAY",
        "HEI": "HEICO",
        "HEI-A": "HEICO",
        "HEI.A": "HEICO",
        "LEN": "LENNAR",
        "LEN-B": "LENNAR",
        "LEN.B": "LENNAR",
    }
    if ticker_clean in DUAL_CLASS_MAP:
        return f"MAP_{DUAL_CLASS_MAP[ticker_clean]}"

    # 3. Pattern-based rule for share classes like ABC-A, ABC-B or ABC/A
    m = re.match(r"^([A-Z]+)[-./]([A-Z0-9])$", ticker_clean)
    if m:
        return f"SYM_{m.group(1)}"

    return f"TICKER_{ticker_clean}"


# ── Portfolio Selection ───────────────────────────────────────────────────────

def select_portfolio(
    combined_score: pd.Series,
    top_n: int = TOP_N,
    fundamentals: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Return the top-N tickers by combined score, enforcing that each selected
    ticker belongs to a unique company (deduplicating multi-share classes).
    If multiple tickers belong to the same underlying company, only the single
    ticker with the highest Combined Score is retained.
    """
    sorted_scores = combined_score.dropna().sort_values(ascending=False)
    selected: List[str] = []
    seen_companies: Set[str] = set()

    for ticker in sorted_scores.index:
        comp_key = get_company_key(ticker, fundamentals)
        if comp_key not in seen_companies:
            seen_companies.add(comp_key)
            selected.append(ticker)
            if len(selected) == top_n:
                break

    return selected


# ── Monthly Signal Generation ─────────────────────────────────────────────────

def generate_monthly_signals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe_tickers: List[str],
    spy_prices: pd.Series,
    bond_ticker: str = BOND_TICKER,
    cash_ticker: str = CASH_TICKER,
    top_n: int = TOP_N,
) -> List[dict]:
    """
    Iterate over every month-end in the price history and generate
    a rebalancing signal dict.

    Each signal is to be APPLIED at the open of the first trading day
    of the NEXT month.

    Returns a list of dicts:
        {
            'signal_date':   last trading day of the signal month,
            'execute_date':  first trading day of the next month (for reference),
            'risk_on':       bool,
            'equity_tickers': list[str],  # selected equity names
            'equity_weight':  float,      # weight per equity position
            'bond_weight':    float,      # TLT weight (0 when risk-on)
            'cash_weight':    float,      # SHV weight (currently unused)
        }
    """
    # --- Build monthly SPY close series ---
    spy_monthly = spy_prices.resample("ME").last()   # month-end close
    macro_signal = compute_macro_signal(spy_monthly)

    # --- Find all month-end dates in the price data ---
    month_ends = prices.resample("ME").last().index
    month_ends = month_ends[month_ends >= prices.index[0]]

    signals = []
    for me_date in month_ends:
        if me_date not in macro_signal.index or pd.isna(macro_signal[me_date]):
            continue

        # Apply quality filter for this rebalance date
        qualified = apply_quality_filter(universe_tickers, fundamentals, as_of_date=me_date)
        if qualified is None:
            qualified = list(universe_tickers)
        # Remove defensive tickers from the equity universe
        equity_universe = [t for t in qualified if t not in (bond_ticker, cash_ticker, "SPY")]
        if me_date not in macro_signal.index or pd.isna(macro_signal[me_date]):
            continue

        risk_on = bool(macro_signal[me_date])

        # Factor scoring on data up to me_date
        available_equity = [t for t in equity_universe if t in prices.columns]
        factor_df = compute_factor_details(prices, available_equity, me_date, fundamentals=fundamentals)

        if factor_df.empty:
            continue

        combined  = factor_df["combined_score"]
        selected  = select_portfolio(combined, top_n=top_n, fundamentals=fundamentals)
        n = len(selected)
        if n == 0:
            continue

        if risk_on:
            eq_weight   = 1.0 / n
            bond_weight = 0.0
        else:
            eq_weight   = DEFENSIVE_EQUITY_ALLOC / n
            bond_weight = DEFENSIVE_BOND_ALLOC

        signals.append({
            "signal_date":    me_date,
            "risk_on":        risk_on,
            "equity_tickers": selected,
            "equity_weight":  eq_weight,
            "bond_weight":    bond_weight,
            "cash_weight":    0.0,
            "bond_ticker":    bond_ticker,
            # Full ranked factor table for reporting / audit
            "factor_details": factor_df,
        })

    logger.info("Generated %d monthly signals.", len(signals))
    return signals

