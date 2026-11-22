"""
config.py — Central configuration for The Value Hunter backtest.
"""

# ── Date range ────────────────────────────────────────────────────────────────
# DATA_START_DATE: how far back to download prices (warmup window for factors).
# Momentum needs 13 months of history before the first signal date (2005-01-31),
# so we start downloading from 2003-12-01.
DATA_START_DATE = "2003-12-01"

# Backtest evaluation period — metrics and returns are reported from here.
START_DATE = "2005-01-01"
END_DATE   = "2026-10-31"

# ── Macro overlay ─────────────────────────────────────────────────────────────
MARKET_TICKER        = "SPY"
MACRO_SMA_MONTHS     = 10          # 10-month SMA look-back
BOND_TICKER          = "TLT"       # defensive bond leg
CASH_TICKER          = "SHV"       # defensive cash leg
DEFENSIVE_BOND_ALLOC = 0.50        # fraction in bonds when risk-off
DEFENSIVE_EQUITY_ALLOC = 0.50      # remaining fraction stays in equities (risk-off)

# ── Universe ──────────────────────────────────────────────────────────────────
# Nasdaq-100 tickers as of mid-2024 (survivorship-bias-aware approach:
# we use a static list; for production use point-in-time constituents).
# SPY / NDX proxies — a representative large-cap growth universe.
UNIVERSE_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "GOOG", "AMZN",
    "TSLA", "ADBE", "ORCL", "CRM", "AMD", "QCOM", "INTC", "TXN",
    "AMAT", "LRCX", "KLAC", "MU", "MRVL", "ON", "NXPI", "MCHP",
    "CDNS", "SNPS", "FTNT", "PANW", "CRWD", "ZS", "OKTA",
    "DDOG", "SNOW", "MDB", "NET", "HUBS", "TEAM", "WDAY", "NOW",
    "INTU", "ADSK", "CTSH", "FISV", "GPN", "FIS",
    # Consumer / E-commerce
    "COST", "AMZN", "EBAY", "BKNG", "EXPE", "ABNB", "DASH",
    "NFLX", "SPOT", "ROKU", "TTD", "PINS", "SNAP", "MTCH",
    # Health / Biotech
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "IDXX",
    "ALGN", "DXCM", "ISRG", "PODD",
    # Industrials / Other large-cap
    "HON", "GE", "CAT", "DE", "MMM", "LMT", "RTX", "NOC", "BA",
    "CSX", "FAST", "ODFL", "VRSK", "CSGP", "CPRT",
    # Financials
    "V", "MA", "PYPL", "AXP", "JPM", "BAC", "GS", "MS",
    # Energy / Materials
    "XOM", "CVX", "LIN", "APD",
    # Communications
    "CMCSA", "TMUS", "CHTR", "EA", "TTWO",
]
# De-duplicate while preserving order
_seen = set()
UNIVERSE_TICKERS = [t for t in UNIVERSE_TICKERS if not (t in _seen or _seen.add(t))]

# ── Quality filters & Outlier / Value-Trap Protections ───────────────────────
FCF_MIN            = 0.0     # exclude negative TTM FCF
DE_BOTTOM_PCT      = 0.20    # exclude bottom 20% by D/E ratio
MOMENTUM_MIN       = 0.0     # value-trap floor: strictly exclude negative 12M-1M return
MAX_FCF_YIELD      = 0.20    # cap abnormal / distressed FCF yield outliers at 20.0%
WINSORIZE_LOWER    = 0.05    # 5th percentile winsorization bound for raw factors
WINSORIZE_UPPER    = 0.95    # 95th percentile winsorization bound for raw factors
ZSCORE_CLIP        = 2.5     # cap cross-sectional Z-scores to [-2.5, +2.5]

# ── Factor weights ────────────────────────────────────────────────────────────
MOMENTUM_WEIGHT    = 0.30
STABILITY_WEIGHT   = 0.30
VALUE_WEIGHT       = 0.40

MOMENTUM_MONTHS    = 12      # look-back months for momentum
MOMENTUM_SKIP      = 1       # skip last month (12M-1M momentum)
STABILITY_DAYS     = 90      # rolling days for price-std (stability)

# ── Portfolio construction ────────────────────────────────────────────────────
TOP_N              = 10      # top-N stocks to hold (concentrated model)
EQUAL_WEIGHT       = 1.0 / TOP_N   # 10% each (0.10)

# ── Capital ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 100_000.0

# ── Transaction Costs & Friction ──────────────────────────────────────────────
COMMISSION_RATE    = 0.0005  # 0.05% per trade
SLIPPAGE_RATE      = 0.0005  # 0.05% on rebalancing prices

# ── Alpaca Credentials & Execution ───────────────────────────────────────────
ALPACA_API_KEY     = None  # Fallback if not set in environment or .env
ALPACA_SECRET_KEY  = None  # Fallback if not set in environment or .env
ALPACA_ENDPOINT    = "https://paper-api.alpaca.markets"  # Paper trading URL
ALPACA_PAPER       = True  # True for paper trading endpoint

# ── Misc ──────────────────────────────────────────────────────────────────────
CACHE_DIR          = "cache"
DATA_DIR           = "data"
TRADING_DAYS_YEAR  = 252

