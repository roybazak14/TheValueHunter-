"""
api_server.py — FastAPI Telemetry, Authentication & Holdings Backend for The Value Hunter.

Features:
  1. SQLite user management (`fund_users.db`) with salted SHA-256/PBKDF2 password hashing.
  2. Automatic seeding of the Admin account (admin@example.com); password comes from
     the ADMIN_PASSWORD environment variable and is stored only as a PBKDF2-HMAC-SHA256 hash.
  3. Authentication & Admin Approval Workflow:
     - POST /api/auth/register
     - POST /api/auth/login
     - GET  /api/admin/users
     - POST /api/admin/update-status
  4. Protected Investor Endpoint:
     - GET  /api/portfolio/holdings (live Alpaca 20-position breakdown for approved users)
  5. Public Telemetry & Dashboard Hosting:
     - GET  /api/live-stats
     - GET  / (serves index.html via FileResponse)
"""

import os
import sys
import time
import hmac
import json
import base64
import secrets
import hashlib
import sqlite3
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import yfinance as yf

# Try importing environment loader and credentials from alpaca_execution or config
try:
    from alpaca_execution import get_alpaca_credentials, init_trading_client, fetch_current_prices
except ImportError:
    def get_alpaca_credentials():
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT, ALPACA_PAPER
        return ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT, ALPACA_PAPER

    def init_trading_client(dry_run: bool = False):
        api_key, secret_key, endpoint, paper = get_alpaca_credentials()
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=endpoint)

    def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}
        px: Dict[str, float] = {}
        try:
            d = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
            c = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d
            for t in tickers:
                if t in c.columns:
                    s = c[t].dropna()
                    if not s.empty:
                        px[t] = float(s.iloc[-1])
        except Exception:
            pass
        return px

try:
    from config import INITIAL_CAPITAL
except ImportError:
    INITIAL_CAPITAL = 100_000.0

# Setup Logger
logger = logging.getLogger("api_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Writable data directory for SQLite / runtime files.
# Locally this defaults to the project root; in the cloud point VH_DATA_DIR at a
# persistent volume (e.g. Render Disk mounted at /var/data) so data survives redeploys.
DATA_DIR = os.path.abspath(os.getenv("VH_DATA_DIR", BASE_DIR))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.abspath(os.getenv("VH_DB_PATH", os.path.join(DATA_DIR, "fund_users.db")))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
HISTORY_CSV_PATH = os.path.join(DATA_DIR, "live_portfolio_history.csv")
SIGNALS_CSV_PATH = os.path.join(BASE_DIR, "current_portfolio_signals.csv")
SIGNALS_JSON_PATH = os.path.join(BASE_DIR, "current_portfolio_signals.json")
MACRO_JSON_PATH = os.path.join(BASE_DIR, "current_macro_regime.json")
BACKTEST_JSON_PATH = os.path.join(BASE_DIR, "backtest_results.json")
SMA_10M_THRESHOLD = 725.56
_LAST_KNOWN_SPY_PRICE = 755.72

def calculate_macro_regime(spy_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate real-time Macro Regime metrics dynamically from latest SPY market price.
    Calculates:
      - spy_price: latest traded price of SPY
      - sma_threshold: static/monthly 10M SMA threshold ($725.56)
      - buffer_pct: ((spy_price - sma_threshold) / sma_threshold) * 100 (rounded to 2 decimals)
      - regime: 'Risk-ON' if spy_price >= sma_threshold else 'Risk-OFF'
    """
    global _LAST_KNOWN_SPY_PRICE
    if spy_price is not None and float(spy_price) > 0:
        _LAST_KNOWN_SPY_PRICE = round(float(spy_price), 2)

    current_spy = _LAST_KNOWN_SPY_PRICE
    sma = float(SMA_10M_THRESHOLD)
    buffer_pct = round(((current_spy - sma) / sma) * 100.0, 2)
    regime = "Risk-ON" if current_spy >= sma else "Risk-OFF"
    is_risk_on = (regime == "Risk-ON")
    regime_label = "Risk-ON (100% Equities)" if is_risk_on else "Risk-OFF (Defense / Bonds)"
    target_alloc = "100% Equities" if is_risk_on else "50% Bonds + 50% Equities"
    buffer_text = f"{'+' if buffer_pct >= 0 else ''}{buffer_pct:.2f}% {'above' if buffer_pct >= 0 else 'below'} SMA safety threshold"

    return {
        "spy_price": current_spy,
        "sma_threshold": sma,
        "spy_10m_sma": sma,
        "buffer_pct": buffer_pct,
        "distance_pct": buffer_pct,
        "regime": regime,
        "status": regime,
        "regime_label": regime_label,
        "target_allocation": target_alloc,
        "equity_allocation_pct": 100.0 if is_risk_on else 50.0,
        "bond_allocation_pct": 0.0 if is_risk_on else 50.0,
        "buffer_text": buffer_text,
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def get_current_macro_regime(spy_price: Optional[float] = None) -> Dict[str, Any]:
    """Retrieve the latest Macro Regime metric (SPY price, 10M SMA, distance %, status)."""
    if spy_price is not None and float(spy_price) > 0:
        return calculate_macro_regime(spy_price)

    if os.path.isfile(MACRO_JSON_PATH):
        try:
            with open(MACRO_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and "spy_price" in data:
                    return calculate_macro_regime(float(data["spy_price"]))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", MACRO_JSON_PATH, exc)

    return calculate_macro_regime()

def load_factors_lookup() -> Dict[str, Dict[str, Any]]:
    """Load factor metrics and 1-sentence rationales for all active holdings."""
    lookup: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(SIGNALS_JSON_PATH):
        try:
            with open(SIGNALS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for s in data.get("signals", []):
                    tk = str(s.get("ticker", "")).strip().upper()
                    if tk:
                        lookup[tk] = {
                            "rank": int(s.get("rank", 0)),
                            "composite_score": float(s.get("combined_score", 0.0)),
                            "fcf_yield": float(s.get("fcf_yield_pct", 0.0)),
                            "volatility_90d": float(s.get("volatility_90d_pct", 0.0)),
                            "momentum_12m_1m": float(s.get("momentum_12m_1m_pct", 0.0)),
                            "z_value": float(s.get("z_value", 0.0)),
                            "z_stability": float(s.get("z_stability", 0.0)),
                            "z_momentum": float(s.get("z_momentum", 0.0)),
                            "rationale": str(s.get("rationale", "")),
                            "rationale_he": str(s.get("rationale_he", s.get("rationale", ""))),
                            "rationale_en": str(s.get("rationale_en", "")),
                            "rationale_summary": str(s.get("rationale_summary", "")),
                            "rationale_summary_en": str(s.get("rationale_summary_en", "")),
                        }
                if lookup:
                    return lookup
        except Exception as exc:
            logger.warning("Failed to load %s: %s", SIGNALS_JSON_PATH, exc)

    if os.path.isfile(SIGNALS_CSV_PATH):
        try:
            df = pd.read_csv(SIGNALS_CSV_PATH)
            for _, r in df.iterrows():
                tk = str(r.get("ticker", "")).strip().upper()
                if tk:
                    lookup[tk] = {
                        "rank": int(r.get("rank", 0)),
                        "composite_score": float(r.get("combined_score", 0.0)),
                        "fcf_yield": float(r.get("fcf_yield_pct", r.get("value_raw", 0.0) * 100.0)),
                        "volatility_90d": float(r.get("volatility_90d_pct", r.get("stability_raw", 0.0) * 100.0)),
                        "momentum_12m_1m": float(r.get("momentum_12m_1m_pct", r.get("momentum_raw", 0.0) * 100.0)),
                        "z_value": float(r.get("z_value", 0.0)),
                        "z_stability": float(r.get("z_stability", 0.0)),
                        "z_momentum": float(r.get("z_momentum", 0.0)),
                        "rationale": str(r.get("rationale", "")),
                        "rationale_he": str(r.get("rationale_he", r.get("rationale", ""))),
                        "rationale_en": str(r.get("rationale_en", "")),
                        "rationale_summary": str(r.get("rationale_summary", "")),
                        "rationale_summary_en": str(r.get("rationale_summary_en", "")),
                    }
        except Exception as exc:
            logger.warning("Failed to load %s: %s", SIGNALS_CSV_PATH, exc)

    return lookup

# In-memory cache for fast historical backtest serving
_BACKTEST_CACHE: Optional[Dict[str, Any]] = None

def load_backtest_cache(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Load or generate cached historical backtest results."""
    global _BACKTEST_CACHE
    if _BACKTEST_CACHE is not None and not force_reload:
        return _BACKTEST_CACHE

    if os.path.isfile(BACKTEST_JSON_PATH):
        try:
            with open(BACKTEST_JSON_PATH, "r", encoding="utf-8") as f:
                _BACKTEST_CACHE = json.load(f)
                logger.info("Loaded pre-computed backtest results from %s", BACKTEST_JSON_PATH)
                return _BACKTEST_CACHE
        except Exception as exc:
            logger.error("Failed to load %s: %s", BACKTEST_JSON_PATH, exc)

    try:
        from generate_backtest_data import generate_backtest_payload
        logger.info("Generating backtest data payload on the fly...")
        _BACKTEST_CACHE = generate_backtest_payload(BACKTEST_JSON_PATH)
        return _BACKTEST_CACHE
    except Exception as exc:
        logger.error("Failed to generate backtest payload: %s", exc)
        return None

INCEPTION_DATE = "2026-10-21"
VH_BASE_EQUITY = 100000.00
SPY_INCEPTION_PRICE = 773.09
QQQ_INCEPTION_PRICE = 725.20

# Default Admin Credentials (seeded automatically on startup).
# Override via environment variables in production.
DEFAULT_ADMIN_EMAIL = os.getenv("ADMIN_EMAIL") or os.getenv("VH_ADMIN_EMAIL", "admin@example.com")
# Read strictly from the environment (set ADMIN_PASSWORD on Render). Only a salted PBKDF2 hash is
# ever written to the database; the plain-text value is never stored or logged.
_ADMIN_PASSWORD_FROM_ENV = os.environ.get("ADMIN_PASSWORD") or os.environ.get("VH_ADMIN_PASSWORD")
DEFAULT_ADMIN_NAME = os.getenv("VH_ADMIN_NAME", "Admin User")
DEFAULT_ADMIN_PHONE = os.getenv("VH_ADMIN_PHONE", "")

# Pre-computed salted PBKDF2-HMAC-SHA256 hash for initial admin seed on fresh instances
_DEFAULT_ADMIN_PBKDF2_HASH = (
    "d55af1ee388e49d2c42c495ca797ba5c$"
    "e190660af777b4eefd8117335b7dbac23248620d526b335759658f2c9ee7dd3d"
)

# List of pre-authorized superusers seeded with admin privileges
ADMIN_SEEDS = [
    {
        "email": DEFAULT_ADMIN_EMAIL.lower(),
        "name": DEFAULT_ADMIN_NAME,
        "phone": DEFAULT_ADMIN_PHONE,
        "password": _ADMIN_PASSWORD_FROM_ENV,
        "password_hash": _DEFAULT_ADMIN_PBKDF2_HASH,
    },
]
ADMIN_EMAILS_SET = { seed["email"].lower() for seed in ADMIN_SEEDS }

# Internal cron authentication token for automated headless triggers
_DEFAULT_TOKEN_SECRET = "value_hunter_institutional_secret_key_2026"
TOKEN_SECRET = os.getenv("VH_TOKEN_SECRET", _DEFAULT_TOKEN_SECRET)
INTERNAL_CRON_TOKEN = os.getenv("INTERNAL_CRON_TOKEN", TOKEN_SECRET)
if TOKEN_SECRET == _DEFAULT_TOKEN_SECRET:
    logger.warning("VH_TOKEN_SECRET is not set; using the insecure built-in default. Set it in production!")
if not _ADMIN_PASSWORD_FROM_ENV:
    logger.warning("ADMIN_PASSWORD / VH_ADMIN_PASSWORD is not set in environment; preserving existing password hash in database.")
logger.info("SQLite database path: %s", DB_PATH)


# ── Password Hashing & Token Utilities ────────────────────────────────────────

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash a password using salted PBKDF2-HMAC-SHA256."""
    if not salt:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored `salt$hash` string."""
    try:
        if "$" not in stored_hash:
            return False
        salt, _ = stored_hash.split("$", 1)
        expected = hash_password(password, salt=salt)
        return hmac.compare_digest(expected, stored_hash)
    except Exception:
        return False


def create_auth_token(user: Dict[str, Any]) -> str:
    """Create an HMAC-SHA256 signed token containing user identity."""
    payload = {
        "uid": user["id"],
        "email": user["email"],
        "role": user["role"],
        "iat": int(datetime.now().timestamp()),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    signature = hmac.new(
        TOKEN_SECRET.encode("utf-8"),
        b64_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64_payload}.{signature}"


def verify_auth_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Verify an HMAC-signed token and return the fresh user record from SQLite."""
    if not token or "." not in token:
        return None
    try:
        b64_payload, signature = token.rsplit(".", 1)
        expected_sig = hmac.new(
            TOKEN_SECRET.encode("utf-8"),
            b64_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None

        padding = "=" * (-len(b64_payload) % 4)
        payload_json = base64.urlsafe_b64decode(b64_payload + padding).decode("utf-8")
        payload = json.loads(payload_json)
        email = str(payload.get("email") or "").strip().lower()
        user_id = payload.get("uid")

        user = None
        if email:
            user = get_user_by_email(email)
            if user:
                user.pop("password_hash", None)
        if not user and user_id is not None:
            user = get_user_by_id(int(user_id))
        if not user and email in ADMIN_EMAILS_SET:
            user = {
                "id": user_id or 1,
                "full_name": DEFAULT_ADMIN_NAME,
                "email": email,
                "phone": DEFAULT_ADMIN_PHONE,
                "role": "admin",
                "status": "approved",
            }
        if user and (
            str(user.get("email") or "").strip().lower() in ADMIN_EMAILS_SET
            or str(payload.get("role") or "").strip().lower() == "admin"
            
        ):
            user["role"] = "admin"
            user["status"] = "approved"
        return user
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
        return None


def extract_request_user(request: Request, token_query: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract authenticated user from Authorization Bearer header, X-Auth-Token, or query param."""
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    elif request.headers.get("X-Auth-Token"):
        token = request.headers.get("X-Auth-Token", "").strip()
    elif token_query and isinstance(token_query, str) and token_query.strip():
        token = token_query.strip()
    elif hasattr(request, "query_params"):
        token = (request.query_params.get("token") or request.query_params.get("auth_token") or "").strip() or None

    return verify_auth_token(token)


# ── SQLite Database Management ────────────────────────────────────────────────

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables (users, email_otps, portfolio_snapshots) and seed the initial admin account and snapshots."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS email_otps (
                    email TEXT,
                    code_hash TEXT,
                    expires_at TIMESTAMP,
                    attempts INTEGER DEFAULT 0
                )
                """
            )

            conn.execute(
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

            # Normalize any existing user emails with leading/trailing whitespace
            conn.execute("UPDATE users SET email = lower(trim(email))")

                        # Normalize any existing user emails with leading/trailing whitespace
            conn.execute("UPDATE users SET email = lower(trim(email))")

            # Delete any duplicate non-admin user records holding the admin email
            conn.execute(
                "DELETE FROM users WHERE lower(trim(email)) = ? AND role != 'admin'",
                (DEFAULT_ADMIN_EMAIL.lower(),)
            )

            # Demote any accounts other than the designated admin to investor
            conn.execute(
                "UPDATE users SET role = 'investor' WHERE role = 'admin' AND lower(email) != ?",
                (DEFAULT_ADMIN_EMAIL.lower(),),
            )

            # Demote any accounts other than the designated admin to investor
            conn.execute(
                "UPDATE users SET role = 'investor' WHERE role = 'admin' AND lower(email) != ?",
                (DEFAULT_ADMIN_EMAIL.lower(),),
            )
            # Ensure secondary accounts are kept as standard approved investors
            conn.execute(
                "UPDATE users SET role = 'investor', status = 'approved' WHERE lower(email) IN ('user@example.com', 'user@example.com') AND lower(email) != ?",
                (DEFAULT_ADMIN_EMAIL.lower(),),
            )

            # Seed / ensure the authorized system administrator (admin@example.com)
            for admin_seed in ADMIN_SEEDS:
                cur = conn.execute(
                    "SELECT id, role, status FROM users WHERE lower(email) = lower(?)",
                    (admin_seed["email"],),
                )
                admin_row = cur.fetchone()
                if admin_row is None:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    seed_pw_hash = (
                        hash_password(admin_seed["password"])
                        if admin_seed.get("password")
                        else admin_seed.get("password_hash", _DEFAULT_ADMIN_PBKDF2_HASH)
                    )
                    conn.execute(
                        """
                        INSERT INTO users (full_name, email, phone, password_hash, role, status, created_at)
                        VALUES (?, ?, ?, ?, 'admin', 'approved', ?)
                        """,
                        (
                            admin_seed["name"],
                            admin_seed["email"],
                            admin_seed["phone"],
                            seed_pw_hash,
                            now_str,
                        ),
                    )
                    logger.info("Seeded admin user: %s (role=admin, status=approved)", admin_seed["email"])
                else:
                    if admin_seed["password"]:
                        conn.execute(
                            "UPDATE users SET role = 'admin', status = 'approved', password_hash = ?, full_name = ?, phone = ? WHERE id = ?",
                            (hash_password(admin_seed["password"]), admin_seed["name"], admin_seed["phone"], admin_row["id"]),
                        )
                        logger.info("Ensured %s is admin with approved status and updated password from environment.", admin_seed["email"])
                    else:
                        conn.execute(
                            "UPDATE users SET role = 'admin', status = 'approved', full_name = ?, phone = ? WHERE id = ?",
                            (admin_seed["name"], admin_seed["phone"], admin_row["id"]),
                        )
                        logger.info("Ensured %s is admin with approved status (preserved existing password hash).", admin_seed["email"])

            # Purge any non-trading days (weekends) from portfolio snapshots
            conn.execute("DELETE FROM portfolio_snapshots WHERE strftime('%w', date) IN ('0', '6') OR date = '2026-11-01'")
            # Normalize Inception date (2026-10-21) baseline to exactly $100,000.00 (0.00%)
            conn.execute(
                """
                UPDATE portfolio_snapshots
                SET portfolio_value = 100000.00,
                    cash = 0.00,
                    positions_value = 100000.00,
                    fund_return_pct = 0.0000,
                    spy_return_pct = 0.0000,
                    qqq_return_pct = 0.0000
                WHERE date = '2026-10-21'
                """
            )

            # Seed existing CSV historical records if portfolio_snapshots is empty
            cur_snap = conn.execute("SELECT COUNT(*) AS cnt FROM portfolio_snapshots")
            if cur_snap.fetchone()["cnt"] == 0:
                _seed_snapshots_from_csv(conn)
    finally:
        conn.close()


def _seed_snapshots_from_csv(conn: sqlite3.Connection) -> None:
    """Import initial portfolio snapshots from CSV if available, filtering out weekends."""
    csv_candidates = [
        HISTORY_CSV_PATH,
        os.path.join(BASE_DIR, "live_portfolio_history.csv"),
    ]
    target_csv = next((p for p in csv_candidates if os.path.isfile(p)), None)
    if not target_csv:
        return
    try:
        df = pd.read_csv(target_csv)
        if df.empty or "date" not in df.columns:
            return
        inserted = 0
        for _, row in df.iterrows():
            d = str(row["date"]).strip()
            # Filter out non-trading days (weekends: Saturday=5, Sunday=6)
            try:
                dt_obj = datetime.strptime(d, "%Y-%m-%d")
                if dt_obj.weekday() >= 5:
                    continue
            except Exception:
                pass

            conn.execute(
                """
                INSERT OR IGNORE INTO portfolio_snapshots
                (date, portfolio_value, cash, positions_value, spy_price, qqq_price, fund_return_pct, spy_return_pct, qqq_return_pct, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'csv_seed')
                """,
                (
                    d,
                    float(row.get("portfolio_value", 0.0)),
                    float(row.get("cash", 0.0)),
                    float(row.get("positions_value", 0.0)),
                    float(row.get("spy_price", SPY_INCEPTION_PRICE)),
                    float(row.get("qqq_price", QQQ_INCEPTION_PRICE)),
                    float(row.get("fund_return_pct", 0.0)),
                    float(row.get("spy_return_pct", 0.0)),
                    float(row.get("qqq_return_pct", 0.0)),
                    str(row.get("updated_at", "")),
                ),
            )
            inserted += 1
        logger.info("Seeded %d portfolio snapshots from %s (excluding weekends)", inserted, target_csv)
    except Exception as exc:
        logger.warning("Could not seed portfolio_snapshots from CSV: %s", exc)


def _export_snapshots_to_csv() -> None:
    """Export all portfolio snapshots from SQLite to live_portfolio_history.csv."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            SELECT date, portfolio_value, cash, positions_value, spy_price, qqq_price,
                   fund_return_pct, spy_return_pct, qqq_return_pct, updated_at
            FROM portfolio_snapshots
            ORDER BY date ASC
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return
        df = pd.DataFrame(rows)
        # Write to DATA_DIR
        df.to_csv(HISTORY_CSV_PATH, index=False)
        # Also sync to BASE_DIR if different
        base_csv = os.path.join(BASE_DIR, "live_portfolio_history.csv")
        if os.path.abspath(base_csv) != os.path.abspath(HISTORY_CSV_PATH):
            try:
                df.to_csv(base_csv, index=False)
            except Exception:
                pass
    except Exception as exc:
        logger.error("Failed to export snapshots to CSV: %s", exc)
    finally:
        conn.close()


def _row_to_user_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "phone": row["phone"],
        "role": row["role"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM users WHERE lower(trim(email)) = lower(trim(?))",
            (email.strip(),),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = _row_to_user_dict(row)
        d["password_hash"] = row["password_hash"]
        return d
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return _row_to_user_dict(row) if row else None
    finally:
        conn.close()


# Ensure DB is initialized on module import
init_db()


# ── Pydantic Request Schemas ──────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str
    email: str
    phone: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RequestOtpPayload(BaseModel):
    email: str


class VerifyOtpPayload(BaseModel):
    email: str
    code: str


class UpdateStatusRequest(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    status: str  # "approved" | "rejected" | "pending"


# ── Extended-Hours Market Session & Quote Helpers ─────────────────────────────

_EXT_QUOTE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "prices": {},
}
EXT_CACHE_TTL_SEC = 8.0

# ── Extensible Broker-Agnostic Price Cache ─────────────────────────────────────
_latest_prices: Dict[str, float] = {}
_LAST_VALID_PORTFOLIO_SNAPSHOT: Optional[Dict[str, Any]] = None


def _get_eastern_now() -> datetime:
    """Return current datetime in US/Eastern (America/New_York)."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            pass
    # Fallback to EDT (-04:00)
    return datetime.now(timezone(timedelta(hours=-4)))


def _ensure_eastern_tz(dt: datetime) -> datetime:
    """Ensure datetime has US/Eastern timezone attached or converted."""
    eastern_tz = None
    if ZoneInfo is not None:
        try:
            eastern_tz = ZoneInfo("America/New_York")
        except Exception:
            pass
    if eastern_tz is None:
        eastern_tz = timezone(timedelta(hours=-4))

    if dt.tzinfo is None:
        return dt.replace(tzinfo=eastern_tz)
    return dt.astimezone(eastern_tz)


def calculate_easter(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher) to find Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th weekday of a month (0=Monday..6=Sunday)."""
    first_day = date(year, month, 1)
    first_target = first_day + timedelta(days=(weekday - first_day.weekday()) % 7)
    return first_target + timedelta(weeks=n - 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of a given weekday in a month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def get_nyse_holidays(year: int) -> Dict[date, str]:
    """
    Official NYSE / Nasdaq Full Market Holidays for a given calendar year:
      1. New Year's Day (Jan 1, observed Jan 2 if Sunday)
      2. Martin Luther King Jr. Day (3rd Mon in Jan)
      3. Washington's Birthday / Presidents' Day (3rd Mon in Feb)
      4. Good Friday (Friday before Easter)
      5. Memorial Day (Last Monday of May)
      6. Juneteenth National Independence Day (June 19, observed)
      7. Independence Day (July 4, observed)
      8. Labor Day (1st Monday of Sep)
      9. Thanksgiving Day (4th Thursday of Nov)
      10. Christmas Day (Dec 25, observed)
    """
    holidays: Dict[date, str] = {}

    # 1. New Year's Day (Jan 1)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 6:  # Sunday -> Monday observed
        holidays[date(year, 1, 2)] = "New Year's Day"
    elif nyd.weekday() != 5:  # If Saturday, NYSE does not observe on preceding Friday unless specified
        holidays[nyd] = "New Year's Day"

    # 2. Martin Luther King Jr. Day (3rd Mon in Jan)
    holidays[_nth_weekday_of_month(year, 1, 0, 3)] = "Martin Luther King Jr. Day"

    # 3. Washington's Birthday / Presidents' Day (3rd Mon in Feb)
    holidays[_nth_weekday_of_month(year, 2, 0, 3)] = "Washington's Birthday / Presidents' Day"

    # 4. Good Friday (Friday before Easter)
    easter = calculate_easter(year)
    holidays[easter - timedelta(days=2)] = "Good Friday"

    # 5. Memorial Day (Last Monday of May)
    holidays[_last_weekday_of_month(year, 5, 0)] = "Memorial Day"

    # 6. Juneteenth National Independence Day (June 19)
    june19 = date(year, 6, 19)
    if june19.weekday() == 6:
        holidays[date(year, 6, 20)] = "Juneteenth National Independence Day"
    elif june19.weekday() == 5:
        holidays[date(year, 6, 18)] = "Juneteenth National Independence Day"
    else:
        holidays[june19] = "Juneteenth National Independence Day"

    # 7. Independence Day (July 4)
    july4 = date(year, 7, 4)
    if july4.weekday() == 6:
        holidays[date(year, 7, 5)] = "Independence Day"
    elif july4.weekday() == 5:
        holidays[date(year, 7, 3)] = "Independence Day"
    else:
        holidays[july4] = "Independence Day"

    # 8. Labor Day (1st Monday of Sep)
    holidays[_nth_weekday_of_month(year, 9, 0, 1)] = "Labor Day"

    # 9. Thanksgiving Day (4th Thursday of Nov)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    holidays[thanksgiving] = "Thanksgiving Day"

    # 10. Christmas Day (Dec 25)
    xmas = date(year, 12, 25)
    if xmas.weekday() == 6:
        holidays[date(year, 12, 26)] = "Christmas Day"
    elif xmas.weekday() == 5:
        holidays[date(year, 12, 24)] = "Christmas Day"
    else:
        holidays[xmas] = "Christmas Day"

    return holidays


def get_nyse_early_close(year: int) -> Dict[date, str]:
    """
    Official NYSE / Nasdaq Early Close Sessions (13:00 ET close):
      1. Day before Independence Day (July 3 if weekday and not full holiday)
      2. Black Friday (Day after Thanksgiving)
      3. Christmas Eve (Dec 24 if weekday and not full holiday)
    """
    early_closes: Dict[date, str] = {}
    holidays = get_nyse_holidays(year)

    # 1. Day before Independence Day
    july3 = date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in holidays:
        early_closes[july3] = "Independence Day Eve"

    # 2. Black Friday (Day after Thanksgiving)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    black_friday = thanksgiving + timedelta(days=1)
    if black_friday not in holidays:
        early_closes[black_friday] = "Day After Thanksgiving (Black Friday)"

    # 3. Christmas Eve (Dec 24)
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5 and xmas_eve not in holidays:
        early_closes[xmas_eve] = "Christmas Eve"

    return early_closes


def resolve_market_session(client=None, dt: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Determine the current US equity market session with true 24/5 coverage & NYSE Calendar:
      - "HOLIDAY CLOSED"   : Official full US market holiday (status: "HOLIDAY CLOSED", badge: Amber/Gray)
      - "WEEKEND CLOSED"   : Fri 20:00 ET to Sun 20:00 ET
      - "OVERNIGHT ACTIVE" : Sun 20:00 ET through Fri 20:00 ET during 20:00 – 04:00 ET
      - "PRE-MARKET ACTIVE": Mon-Fri 04:00 – 09:30 ET
      - "MARKET OPEN"      : Mon-Fri 09:30 – 16:00 ET (or 13:00 ET on half-days)
      - "AFTER-HOURS ACTIVE": Mon-Fri 16:00 – 20:00 ET (or 13:00 – 20:00 ET on half-days)

    Returns dict with:
      market_status, session_name, market_phase, session_phase,
      is_market_open, is_regular_open, is_holiday, holiday_name,
      is_early_close, early_close_reason, trade_date, timestamp_str
    """
    market_dt = dt
    is_regular_open = False

    if market_dt is None and client is not None:
        try:
            clock = client.get_clock()
            market_dt = clock.timestamp
            is_regular_open = bool(clock.is_open)
        except Exception as e:
            logger.warning("Could not fetch Alpaca clock (%s), using Eastern time fallback.", e)

    if market_dt is None:
        market_dt = _get_eastern_now()
    else:
        market_dt = _ensure_eastern_tz(market_dt)

    trade_date = market_dt.strftime("%Y-%m-%d")
    timestamp_str = market_dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()

    cur_date = market_dt.date()
    year = cur_date.year
    holidays = get_nyse_holidays(year)
    early_closes = get_nyse_early_close(year)

    is_holiday = cur_date in holidays
    holiday_name = holidays.get(cur_date)

    is_early_close = cur_date in early_closes
    early_close_reason = early_closes.get(cur_date)

    weekday = market_dt.weekday()  # 0=Monday .. 6=Sunday
    minutes = market_dt.hour * 60 + market_dt.minute

    # 1. Holiday Check (Highest Priority)
    if is_holiday:
        market_status = "HOLIDAY CLOSED"
        session_name = "HOLIDAY CLOSED"
        market_phase = "HOLIDAY"
        session_phase = "HOLIDAY"
        is_market_open = False
        is_regular_open = False

    # 2. Weekend Check: Fri 20:00 ET to Sun 20:00 ET
    elif (weekday == 4 and minutes >= 1200) or (weekday == 5) or (weekday == 6 and minutes < 1200):
        market_status = "WEEKEND CLOSED"
        session_name = "WEEKEND CLOSED"
        market_phase = "WEEKEND"
        session_phase = "WEEKEND"
        is_market_open = False
        is_regular_open = False

    # 3. 24/5 Active Sessions (Mon-Fri or Sun >= 20:00 ET)
    else:
        regular_close_minutes = 780 if is_early_close else 960  # 13:00 ET on half days, 16:00 ET normal

        if weekday < 5 and 240 <= minutes < 570:  # Mon-Fri 04:00 – 09:30 ET
            market_status = "PRE_MARKET"
            session_name = "PRE-MARKET ACTIVE"
            market_phase = "PRE"
            session_phase = "PRE"
            is_market_open = True
            is_regular_open = False
        elif weekday < 5 and 570 <= minutes < regular_close_minutes:  # Mon-Fri 09:30 – 16:00 (or 13:00) ET
            market_status = "REGULAR"
            session_name = "MARKET OPEN"
            market_phase = "REGULAR"
            session_phase = "REGULAR"
            is_market_open = True
            is_regular_open = True
        elif weekday < 5 and regular_close_minutes <= minutes < 1200:  # Mon-Fri 16:00 (or 13:00) – 20:00 ET
            market_status = "AFTER_HOURS"
            session_name = "AFTER-HOURS ACTIVE"
            market_phase = "POST"
            session_phase = "POST"
            is_market_open = True
            is_regular_open = False
        else:
            # Overnight session: Sun 20:00 - Mon 04:00, or Mon-Fri 20:00 - 04:00 ET
            market_status = "OVERNIGHT"
            session_name = "OVERNIGHT ACTIVE"
            market_phase = "OVERNIGHT"
            session_phase = "OVERNIGHT"
            is_market_open = True
            is_regular_open = False

    return {
        "market_status": market_status,
        "session_name": session_name,
        "market_phase": market_phase,
        "session_phase": session_phase,
        "is_market_open": is_market_open,
        "is_regular_open": is_regular_open,
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        "is_early_close": is_early_close,
        "early_close_reason": early_close_reason,
        "trade_date": trade_date,
        "timestamp_str": timestamp_str,
    }


def fetch_benchmark_extended_details(symbol: str, market_status: str) -> Dict[str, float]:
    """
    Fetch real-time quote, previous session close (`previousClose`), 1D dollar change,
    and 1D percentage change for a benchmark ticker (SPY / QQQ).
    """
    tk = yf.Ticker(symbol)
    current_price = None
    previous_close = None

    try:
        info = tk.info or {}
        prev_c = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if prev_c is not None and float(prev_c) > 0:
            previous_close = float(prev_c)

        if market_status == "PRE_MARKET":
            pre_price = info.get("preMarketPrice")
            if pre_price is not None and float(pre_price) > 0:
                current_price = float(pre_price)
        elif market_status in ("AFTER_HOURS", "OVERNIGHT"):
            post_price = info.get("postMarketPrice")
            if post_price is not None and float(post_price) > 0:
                current_price = float(post_price)
        elif market_status in ("CLOSED", "WEEKEND CLOSED", "HOLIDAY CLOSED"):
            for field in ("postMarketPrice", "preMarketPrice", "regularMarketPrice"):
                val = info.get(field)
                if val is not None and float(val) > 0:
                    current_price = float(val)
                    break

        if current_price is None and market_status in ("PRE_MARKET", "AFTER_HOURS", "OVERNIGHT"):
            hist = tk.history(period="2d", interval="1m", prepost=True)
            if not hist.empty and "Close" in hist.columns:
                valid_closes = hist["Close"].dropna()
                if not valid_closes.empty and float(valid_closes.iloc[-1]) > 0:
                    current_price = float(valid_closes.iloc[-1])

        if current_price is None:
            reg_price = info.get("regularMarketPrice")
            if reg_price is not None and float(reg_price) > 0:
                current_price = float(reg_price)
    except Exception as exc:
        logger.debug("Extended info lookup for %s fell back to fast_info (%s)", symbol, exc)

    if current_price is None:
        try:
            current_price = float(tk.fast_info["last_price"])
        except Exception:
            cached_px = _latest_prices.get(symbol)
            if cached_px is not None and cached_px > 0:
                current_price = float(cached_px)

    if current_price is not None and current_price > 0:
        _latest_prices[symbol] = round(current_price, 4)

    if previous_close is None or previous_close <= 0:
        try:
            fast_prev = getattr(tk.fast_info, "previous_close", None)
            if not fast_prev:
                fast_prev = tk.fast_info["previous_close"]
            if fast_prev and float(fast_prev) > 0:
                previous_close = float(fast_prev)
        except Exception:
            pass

    if previous_close is None or previous_close <= 0 or abs(current_price - previous_close) < 1e-5:
        try:
            hist_1d = tk.history(period="5d", interval="1d")
            if not hist_1d.empty and "Close" in hist_1d.columns:
                closes = hist_1d["Close"].dropna()
                if len(closes) >= 2:
                    previous_close = float(closes.iloc[-2])
        except Exception:
            pass

    if previous_close is None or previous_close <= 0:
        previous_close = current_price

    daily_dollar_change = current_price - previous_close
    daily_return_pct = ((current_price - previous_close) / previous_close) * 100.0 if previous_close > 0 else 0.0

    return {
        "price": round(current_price, 4),
        "previous_close": round(previous_close, 4),
        "daily_dollar_change": round(daily_dollar_change, 4),
        "daily_return_pct": round(daily_return_pct, 4),
    }


def fetch_benchmark_extended_quote(symbol: str, market_status: str) -> float:
    """Backwards-compatible helper returning only the float price."""
    return fetch_benchmark_extended_details(symbol, market_status)["price"]


def fetch_live_spy_price(market_status: str = "REGULAR") -> float:
    """Fetch real-time latest market price of SPY with fast fallback and _latest_prices cache."""
    global _LAST_KNOWN_SPY_PRICE
    try:
        details = fetch_benchmark_extended_details("SPY", market_status)
        if details and details.get("price") and float(details["price"]) > 0:
            _LAST_KNOWN_SPY_PRICE = round(float(details["price"]), 2)
            _latest_prices["SPY"] = _LAST_KNOWN_SPY_PRICE
            return _LAST_KNOWN_SPY_PRICE
    except Exception as exc:
        logger.debug("Live SPY price fetch warning: %s", exc)

    cached_spy = _latest_prices.get("SPY")
    if cached_spy and cached_spy > 0:
        return round(float(cached_spy), 2)

    return _LAST_KNOWN_SPY_PRICE


def fetch_batch_extended_quotes(symbols: List[str], market_status: str) -> Dict[str, float]:
    """
    Fetch real-time latest quotes (including pre-market, after-hours, and overnight trades)
    for a list of portfolio symbols using 1-minute `prepost=True` bars with a short TTL cache.
    Updates the broker-agnostic _latest_prices cache.
    """
    if not symbols:
        return {}

    now_ts = time.time()
    if (now_ts - _EXT_QUOTE_CACHE["timestamp"] <= EXT_CACHE_TTL_SEC) and _EXT_QUOTE_CACHE["prices"]:
        cached_prices = _EXT_QUOTE_CACHE["prices"]
        if all(s in cached_prices for s in symbols):
            return {s: cached_prices[s] for s in symbols if s in cached_prices}

    quotes: Dict[str, float] = {}
    try:
        data = yf.download(
            symbols,
            period="2d",
            interval="1m",
            prepost=True,
            progress=False,
            threads=True,
        )
        if not data.empty:
            if isinstance(data.columns, yf.pd.MultiIndex if hasattr(yf, "pd") else type(data.columns)) and hasattr(data.columns, "levels"):
                close_df = data["Close"] if "Close" in data.columns.levels[0] else data.xs("Close", axis=1, level=1)
                for sym in symbols:
                    if sym in close_df.columns:
                        s_valid = close_df[sym].dropna()
                        if not s_valid.empty and float(s_valid.iloc[-1]) > 0:
                            quotes[sym] = float(s_valid.iloc[-1])
            elif "Close" in data.columns and len(symbols) == 1:
                s_valid = data["Close"].dropna()
                if not s_valid.empty and float(s_valid.iloc[-1]) > 0:
                    quotes[symbols[0]] = float(s_valid.iloc[-1])
    except Exception as exc:
        logger.warning("Batch extended quote fetch warning: %s", exc)

    if quotes:
        _EXT_QUOTE_CACHE["timestamp"] = now_ts
        _EXT_QUOTE_CACHE["prices"].update(quotes)
        _latest_prices.update(quotes)

    return quotes


_PREV_CLOSE_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "prices": {},
}
PREV_CLOSE_CACHE_TTL_SEC = 60.0


def fetch_batch_previous_closes(
    symbols: List[str],
    cur_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Fetch previous trading session close prices for a list of symbols using
    `yf.download(..., period='5d', interval='1d')` with a 60s cache and
    per-ticker `fast_info` fallback.
    """
    if not symbols:
        return {}

    now_ts = time.time()
    if (now_ts - _PREV_CLOSE_CACHE["timestamp"] <= PREV_CLOSE_CACHE_TTL_SEC) and _PREV_CLOSE_CACHE["prices"]:
        cached = _PREV_CLOSE_CACHE["prices"]
        if all(s in cached and cached[s] > 0 for s in symbols):
            return {s: cached[s] for s in symbols}

    prev_closes: Dict[str, float] = {}
    today_et_str = _get_eastern_now().strftime("%Y-%m-%d")

    try:
        data = yf.download(
            symbols,
            period="5d",
            interval="1d",
            progress=False,
            threads=True,
        )
        if not data.empty:
            if isinstance(data.columns, yf.pd.MultiIndex if hasattr(yf, "pd") else type(data.columns)) and hasattr(data.columns, "levels"):
                close_df = data["Close"] if "Close" in data.columns.levels[0] else data.xs("Close", axis=1, level=1)
                for sym in symbols:
                    if sym in close_df.columns:
                        s_valid = close_df[sym].dropna()
                        if len(s_valid) >= 2:
                            last_bar_date = str(s_valid.index[-1])[:10]
                            last_bar_px = float(s_valid.iloc[-1])
                            cur_px = (cur_prices or {}).get(sym, last_bar_px)
                            # If the last daily bar is today's bar or matches cur_price, use iloc[-2] as previous session close
                            if last_bar_date == today_et_str or abs(last_bar_px - cur_px) < 1e-2:
                                prev_closes[sym] = float(s_valid.iloc[-2])
                            else:
                                prev_closes[sym] = float(s_valid.iloc[-1])
                        elif len(s_valid) == 1:
                            prev_closes[sym] = float(s_valid.iloc[-1])
            elif "Close" in data.columns and len(symbols) == 1:
                s_valid = data["Close"].dropna()
                if len(s_valid) >= 2:
                    prev_closes[symbols[0]] = float(s_valid.iloc[-2])
                elif len(s_valid) == 1:
                    prev_closes[symbols[0]] = float(s_valid.iloc[-1])
    except Exception as exc:
        logger.warning("Batch previous close download warning: %s", exc)

    # Individual ticker fallback via fast_info if any symbol is still missing
    for sym in symbols:
        if sym not in prev_closes or prev_closes[sym] <= 0:
            try:
                fi = yf.Ticker(sym).fast_info
                fast_prev = getattr(fi, "previous_close", None)
                if not fast_prev:
                    fast_prev = fi["previous_close"]
                if fast_prev and float(fast_prev) > 0:
                    prev_closes[sym] = float(fast_prev)
            except Exception:
                pass

    if prev_closes:
        _PREV_CLOSE_CACHE["timestamp"] = now_ts
        _PREV_CLOSE_CACHE["prices"].update(prev_closes)

    return prev_closes


# ── Alpaca & Benchmark Helpers ────────────────────────────────────────────────

def compute_positions_daily_changes(
    positions: List[Any], ext_quotes: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Single source of truth for per-position 1D session change.
    Resolves current price (extended-hours quote when available) and previous close
    (Alpaca `lastday_price`, falling back to yfinance when missing/stale), then computes
    per-position daily $ / % change. Used by both the KPI card (/api/live-stats) and the
    holdings table (/api/portfolio/holdings) so the two always reconcile.
    """
    cur_prices_map: Dict[str, float] = {}
    needs_yf_prev = False
    for p in positions:
        sym = str(p.symbol)
        live_px = ext_quotes.get(sym)
        c_px = float(live_px) if (live_px is not None and live_px > 0) else float(p.current_price)
        cur_prices_map[sym] = c_px
        raw_lastday = getattr(p, "lastday_price", None)
        ld_px = float(raw_lastday) if (raw_lastday is not None and float(raw_lastday) > 0) else 0.0
        if ld_px <= 0 or abs(c_px - ld_px) < 1e-4:
            needs_yf_prev = True

    yf_prev_closes: Dict[str, float] = {}
    if needs_yf_prev and cur_prices_map:
        yf_prev_closes = fetch_batch_previous_closes(list(cur_prices_map.keys()), cur_prices_map)

    results: Dict[str, Dict[str, float]] = {}
    for p in positions:
        sym = str(p.symbol)
        qty = float(p.qty)
        cur_price = cur_prices_map[sym]

        raw_lastday = getattr(p, "lastday_price", None)
        prev_close = float(raw_lastday) if (raw_lastday is not None and float(raw_lastday) > 0) else 0.0
        if prev_close <= 0 or abs(cur_price - prev_close) < 1e-4:
            yf_prev = yf_prev_closes.get(sym)
            if yf_prev is not None and yf_prev > 0:
                prev_close = float(yf_prev)

        if prev_close > 0:
            stock_daily_change_dollar = cur_price - prev_close
            daily_change_percent = (stock_daily_change_dollar / prev_close) * 100.0
            daily_change_dollar = qty * stock_daily_change_dollar
        else:
            raw_chg_today = float(getattr(p, "change_today", 0.0) or 0.0)
            daily_change_percent = raw_chg_today * 100.0
            daily_change_dollar = float(getattr(p, "unrealized_intraday_pl", 0.0) or 0.0)
            stock_daily_change_dollar = (daily_change_dollar / qty) if qty > 0 else 0.0

        results[sym] = {
            "cur_price": cur_price,
            "prev_close": prev_close,
            "daily_change_dollar": daily_change_dollar,
            "daily_change_percent": daily_change_percent,
            "stock_daily_change_dollar": stock_daily_change_dollar,
        }
    return results


_TODAY_FILLS_CACHE: Dict[str, Any] = {"timestamp": 0.0, "date": None, "fills": None}
TODAY_FILLS_CACHE_TTL_SEC = 15.0
_CLOSED_PREV_CLOSE_CACHE: Dict[str, Any] = {"date": None, "prices": {}}


def fetch_today_fills(client, session_start_et: datetime) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Aggregate today's filled orders per symbol since the start of the current ET trading day:
      {symbol: {"buy_qty", "buy_cost", "sell_qty", "sell_proceeds"}}
    Returns None if orders cannot be retrieved (caller falls back to prev-close method).
    Cached for TODAY_FILLS_CACHE_TTL_SEC to keep 10s dashboard polling light.
    """
    date_key = session_start_et.strftime("%Y-%m-%d")
    now_ts = time.time()
    if (
        _TODAY_FILLS_CACHE["date"] == date_key
        and _TODAY_FILLS_CACHE["fills"] is not None
        and now_ts - _TODAY_FILLS_CACHE["timestamp"] <= TODAY_FILLS_CACHE_TTL_SEC
    ):
        return _TODAY_FILLS_CACHE["fills"]

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        start_utc = session_start_et.astimezone(timezone.utc)
        orders = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=start_utc, limit=500)
        )
    except Exception as exc:
        logger.warning("Could not fetch today's filled orders for 1D reconciliation: %s", exc)
        return None

    fills: Dict[str, Dict[str, float]] = {}
    start_utc = session_start_et.astimezone(timezone.utc)
    for o in orders or []:
        try:
            fq = float(o.filled_qty or 0)
            fp = float(o.filled_avg_price or 0)
            filled_at = getattr(o, "filled_at", None)
            if fq <= 0 or fp <= 0 or filled_at is None:
                continue
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
            if filled_at < start_utc:
                continue
            sym = str(o.symbol)
            rec = fills.setdefault(sym, {"buy_qty": 0.0, "buy_cost": 0.0, "sell_qty": 0.0, "sell_proceeds": 0.0})
            if "buy" in str(o.side).lower():
                rec["buy_qty"] += fq
                rec["buy_cost"] += fq * fp
            else:
                rec["sell_qty"] += fq
                rec["sell_proceeds"] += fq * fp
        except Exception as row_exc:
            logger.debug("Skipping unparsable order in fills aggregation: %s", row_exc)

    _TODAY_FILLS_CACHE.update({"timestamp": now_ts, "date": date_key, "fills": fills})
    return fills


def _closed_symbols_prev_closes(
    symbols: List[str], date_key: str, sell_px_hint: Dict[str, float]
) -> Dict[str, float]:
    """Previous-session closes for symbols fully liquidated today (cached per trading day)."""
    if _CLOSED_PREV_CLOSE_CACHE["date"] != date_key:
        _CLOSED_PREV_CLOSE_CACHE.update({"date": date_key, "prices": {}})
    cache = _CLOSED_PREV_CLOSE_CACHE["prices"]
    missing = [s_ for s_ in symbols if s_ not in cache or cache[s_] <= 0]
    if missing:
        # Pass the sell fill price as the "current" hint so the helper decides on bar date, not price equality
        fetched = fetch_batch_previous_closes(missing, {s_: sell_px_hint.get(s_, -1.0) for s_ in missing})
        cache.update({k: v for k, v in fetched.items() if v and v > 0})
    return {s_: cache.get(s_, 0.0) for s_ in symbols}


def reconcile_daily_pnl(
    client,
    positions: List[Any],
    per_position: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Rebalance-aware 1D P&L using the cash-flow identity per symbol:

        day_pnl = Q_now * P_now - Q_prev * P_prev_close - buy_cost_today + sell_proceeds_today
        where Q_prev = Q_now - buy_qty_today + sell_qty_today

    - Carried positions (no fills today):  Q * (P_now - P_prev_close)           (classic 1D change)
    - Opened today:                        Q * (P_now - avg_fill_price)         (from execution price)
    - Topped-up / trimmed:                 blended correctly by the identity
    - Fully closed today:                  realized vs prev close (+ intraday)  (kept in the total)

    Summed across symbols this equals Δ(positions value) − net trade cash, i.e. Alpaca's
    `equity − last_equity` (excluding fees/dividends/transfers).
    Mutates `per_position` rows in place with reconciled daily_change_dollar / daily_change_percent
    and basis metadata, and returns realized/closed-position details.
    """
    session_start_et = _get_eastern_now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_key = session_start_et.strftime("%Y-%m-%d")
    fills = fetch_today_fills(client, session_start_et)

    if fills is None:
        for row in per_position.values():
            row.setdefault("basis", "prev_close")
            row.setdefault("opened_today", False)
        return {"reconciled": False, "realized_closed_today": 0.0, "closed_today": []}

    open_syms = {str(p.symbol): float(p.qty) for p in positions}

    for sym, q_now in open_syms.items():
        row = per_position.get(sym)
        if row is None:
            continue
        f = fills.get(sym)
        if not f:
            row["basis"] = "prev_close"
            row["opened_today"] = False
            continue

        p_now = row["cur_price"]
        p_prev = row["prev_close"]
        q_prev = q_now - f["buy_qty"] + f["sell_qty"]
        if abs(q_prev) < 1e-6:
            q_prev = 0.0

        day_pnl = q_now * p_now - q_prev * p_prev - f["buy_cost"] + f["sell_proceeds"]
        capital_base = q_prev * p_prev + f["buy_cost"]
        row["daily_change_dollar"] = day_pnl
        row["daily_change_percent"] = (day_pnl / capital_base) * 100.0 if capital_base > 0 else 0.0
        row["opened_today"] = q_prev == 0.0
        row["basis"] = "fill_price" if q_prev == 0.0 else "blended"
        row["avg_fill_price_today"] = (f["buy_cost"] / f["buy_qty"]) if f["buy_qty"] > 0 else None

    closed_syms = [s_ for s_ in fills if s_ not in open_syms]
    closed_details: List[Dict[str, Any]] = []
    realized_total = 0.0
    if closed_syms:
        sell_hint = {
            s_: (fills[s_]["sell_proceeds"] / fills[s_]["sell_qty"]) if fills[s_]["sell_qty"] > 0 else -1.0
            for s_ in closed_syms
        }
        prev_map = _closed_symbols_prev_closes(closed_syms, date_key, sell_hint)
        for sym in closed_syms:
            f = fills[sym]
            q_prev = f["sell_qty"] - f["buy_qty"]
            if abs(q_prev) < 1e-6:
                q_prev = 0.0
            p_prev = prev_map.get(sym, 0.0)
            if q_prev > 0 and p_prev <= 0:
                logger.warning("Missing previous close for closed symbol %s; realized day P&L excluded.", sym)
                continue
            pnl = -q_prev * p_prev - f["buy_cost"] + f["sell_proceeds"]
            pnl_r = round(pnl, 2)
            realized_total += pnl_r
            closed_details.append({
                "symbol": sym,
                "qty_sold": round(f["sell_qty"], 6),
                "avg_sell_price": round(sell_hint[sym], 4) if sell_hint[sym] > 0 else None,
                "previous_close": round(p_prev, 4),
                "realized_day_pnl": pnl_r,
            })
        closed_details.sort(key=lambda d: d["symbol"])

    return {
        "reconciled": True,
        "realized_closed_today": round(realized_total, 2),
        "closed_today": closed_details,
    }


def aggregate_holdings_daily_change(
    per_position: Dict[str, Dict[str, float]],
    current_equity: float,
    realized_closed_today: float = 0.0,
) -> Tuple[float, float]:
    """
    Aggregate 1D session change:
      total_daily_dollar  = sum(round(open_position.daily_change_dollar, 2)) + realized_closed_today
      base_prev_equity    = current_equity - total_daily_dollar
      total_daily_pct     = total_daily_dollar / base_prev_equity * 100
    Sums the rounded per-row values so the KPI card matches the table to the cent.
    """
    total_daily_dollar = round(
        sum(round(v["daily_change_dollar"], 2) for v in per_position.values())
        + round(realized_closed_today, 2),
        2,
    )
    base_prev_equity = current_equity - total_daily_dollar
    total_daily_pct = (total_daily_dollar / base_prev_equity) * 100.0 if base_prev_equity > 0 else 0.0
    return total_daily_dollar, total_daily_pct


def compute_account_daily_change(account: Any) -> Dict[str, float]:
    """
    1D session change straight from the Alpaca account object:
      daily_change_dollar = equity - last_equity   (last_equity = previous trading day close equity)
      daily_change_pct    = daily_change_dollar / last_equity * 100
    """
    current_equity = float(account.equity)
    last_equity_raw = getattr(account, "last_equity", None)
    last_equity = float(last_equity_raw) if last_equity_raw else current_equity
    daily_change_dollar = round(current_equity - last_equity, 2)
    daily_change_pct = round((daily_change_dollar / last_equity) * 100, 2) if last_equity else 0.0
    return {
        "current_equity": current_equity,
        "last_equity": last_equity,
        "daily_change_dollar": daily_change_dollar,
        "daily_change_pct": daily_change_pct,
    }


def fetch_alpaca_portfolio() -> Dict[str, Any]:
    """
    Connect to Alpaca Trading API and fetch current portfolio status.
    Recalculates position market values and total equity during extended hours
    using real-time quotes where available.
    """
    client = init_trading_client(dry_run=False)
    account = client.get_account()

    # Portfolio value is bound strictly to Alpaca's account equity (positions + cash).
    daily = compute_account_daily_change(account)
    equity = daily["current_equity"]
    last_equity = daily["last_equity"]
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    session_info = resolve_market_session(client)
    market_status = session_info["market_status"]
    is_open = session_info["is_market_open"]
    trade_date = session_info["trade_date"]
    timestamp_str = session_info["timestamp_str"]

    positions = client.get_all_positions()
    symbols = [str(p.symbol) for p in positions]

    # During extended/overnight sessions, use live extended quotes for the positions value display
    ext_quotes: Dict[str, float] = {}
    if market_status in ("PRE_MARKET", "AFTER_HOURS", "OVERNIGHT") and symbols:
        ext_quotes = fetch_batch_extended_quotes(symbols, market_status)
        for sym, px in ext_quotes.items():
            if px and px > 0:
                _latest_prices[sym] = float(px)

    positions_value = 0.0
    for p in positions:
        sym = str(p.symbol)
        qty = float(p.qty)
        live_px = ext_quotes.get(sym) or _latest_prices.get(sym)
        if live_px is not None and live_px > 0:
            positions_value += qty * live_px
            _latest_prices[sym] = float(live_px)
        else:
            positions_value += float(p.market_value)
            cur_p = float(getattr(p, "current_price", 0.0) or 0.0)
            if cur_p > 0:
                _latest_prices[sym] = cur_p

    fund_return_pct = round(((equity / VH_BASE_EQUITY) - 1.0) * 100.0, 4)

    # 1D session change = Alpaca equity - last_equity (previous trading day close equity)
    daily_dollar_change = daily["daily_change_dollar"]
    daily_return_pct = daily["daily_change_pct"]

    return {
        "total_equity": round(equity, 2),
        "equity": round(equity, 2),
        "portfolio_value": round(equity, 2),
        "vh_base_equity": VH_BASE_EQUITY,
        "fund_return_pct": fund_return_pct,
        "last_equity": round(last_equity, 2),
        "base_prev_equity": round(last_equity, 2),
        "daily_change_dollar": daily_dollar_change,
        "daily_change_pct": daily_return_pct,
        "daily_dollar_change": daily_dollar_change,
        "daily_return_pct": daily_return_pct,
        "daily_change_source": "alpaca_last_equity",
        "account_daily_dollar_change": daily_dollar_change,
        "account_daily_return_pct": daily_return_pct,
        "cash_balance": round(cash, 2),
        "cash": round(cash, 2),
        "buying_power": round(buying_power, 2),
        "positions_value": round(positions_value, 2),
        "positions_count": len(positions),
        "account_number": str(account.account_number),
        "account_status": str(account.status),
        "trade_date": trade_date,
        "timestamp": timestamp_str,
        "is_market_open": is_open,
        "is_regular_open": session_info["is_regular_open"],
        "market_status": market_status,
        "market_phase": session_info["market_phase"],
        "session_phase": session_info["session_phase"],
        "session_name": session_info["session_name"],
        "is_holiday": session_info["is_holiday"],
        "holiday_name": session_info["holiday_name"],
        "is_early_close": session_info["is_early_close"],
        "early_close_reason": session_info["early_close_reason"],
    }


def fetch_alpaca_holdings() -> Dict[str, Any]:
    """
    Fetch full position-level breakdown for all active Alpaca positions.
    During extended trading sessions (PRE_MARKET / AFTER_HOURS), recalculates
    current_price, market_value, unrealized_pl, unrealized_plpc, and 1D change
    using real-time quotes.
    """
    client = init_trading_client(dry_run=False)
    account = client.get_account()
    cash = float(account.cash) if getattr(account, "cash", None) else 0.0
    # Total equity and 1D change come straight from the Alpaca account object.
    daily = compute_account_daily_change(account)
    total_equity = daily["current_equity"]
    last_equity = daily["last_equity"]

    session_info = resolve_market_session(client)
    market_status = session_info["market_status"]

    positions = client.get_all_positions()
    symbols = [str(p.symbol) for p in positions]

    ext_quotes: Dict[str, float] = {}
    if market_status in ("PRE_MARKET", "AFTER_HOURS", "OVERNIGHT") and symbols:
        ext_quotes = fetch_batch_extended_quotes(symbols, market_status)
        for sym, px in ext_quotes.items():
            if px and px > 0:
                _latest_prices[sym] = float(px)

    # Shared per-position 1D change engine (identical to the KPI card computation),
    # reconciled against today's fills so rebalance days reflect true economic P&L
    per_position = compute_positions_daily_changes(positions, ext_quotes)
    recon = reconcile_daily_pnl(client, positions, per_position)

    raw_rows: List[Dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_unrealized_pl = 0.0

    for p in positions:
        sym = str(p.symbol)
        qty = float(p.qty)
        avg_entry = float(p.avg_entry_price)
        cost_basis = float(getattr(p, "cost_basis", qty * avg_entry))

        live_px = ext_quotes.get(sym) or _latest_prices.get(sym)
        if live_px is not None and live_px > 0:
            cur_price = float(live_px)
            mkt_val = qty * cur_price
            unrealized_pl = mkt_val - cost_basis
            unrealized_plpc_raw = (unrealized_pl / cost_basis) if cost_basis > 0 else 0.0
            _latest_prices[sym] = cur_price
        else:
            cur_price = float(p.current_price)
            mkt_val = float(p.market_value)
            unrealized_pl = float(p.unrealized_pl)
            unrealized_plpc_raw = float(p.unrealized_plpc)
            if cur_price > 0:
                _latest_prices[sym] = cur_price

        unrealized_plpc_pct = unrealized_plpc_raw * 100.0

        # Per-position 1D change from the shared engine
        chg = per_position[sym]
        prev_close = chg["prev_close"]
        daily_change_dollar = chg["daily_change_dollar"]
        daily_change_percent = chg["daily_change_percent"]
        stock_daily_change_dollar = chg["stock_daily_change_dollar"]

        total_market_value += mkt_val
        total_cost_basis += cost_basis
        total_unrealized_pl += unrealized_pl

        raw_rows.append({
            "symbol": sym,
            "qty": round(qty, 4),
            "avg_entry_price": round(avg_entry, 2),
            "current_price": round(cur_price, 2),
            "previous_close": round(prev_close, 2),
            "daily_change_dollar": round(daily_change_dollar, 2),
            "daily_change_percent": round(daily_change_percent, 2),
            "stock_daily_change_dollar": round(stock_daily_change_dollar, 2),
            "price_change_percent": round(
                ((cur_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0, 2
            ),
            "opened_today": bool(chg.get("opened_today", False)),
            "daily_basis": chg.get("basis", "prev_close"),
            "avg_fill_price_today": (
                round(chg["avg_fill_price_today"], 4) if chg.get("avg_fill_price_today") else None
            ),
            "market_value": round(mkt_val, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_plpc": round(unrealized_plpc_pct, 2),
            "unrealized_plpc_raw": round(unrealized_plpc_raw, 6),
            "side": str(getattr(p, "side", "long")).replace("PositionSide.", "").lower(),
        })

    holdings_list: List[Dict[str, Any]] = []
    factors_lookup = load_factors_lookup()
    for row in raw_rows:
        weight_pct = (row["market_value"] / total_equity * 100.0) if total_equity > 0 else 0.0
        row["weight_pct"] = round(weight_pct, 2)
        row["factors"] = factors_lookup.get(str(row["symbol"]).strip().upper(), None)
        holdings_list.append(row)

    # Sort positions by market value descending
    holdings_list.sort(key=lambda x: x["market_value"], reverse=True)

    total_unrealized_plpc = (
        (total_unrealized_pl / total_cost_basis) * 100.0 if total_cost_basis > 0 else 0.0
    )

    # Portfolio-level 1D change = Alpaca equity - last_equity (identical to the KPI card)
    total_daily_change_dollar = daily["daily_change_dollar"]
    total_daily_change_percent = daily["daily_change_pct"]

    payload = {
        "market_status": market_status,
        "market_phase": session_info["market_phase"],
        "session_phase": session_info["session_phase"],
        "session_name": session_info["session_name"],
        "is_holiday": session_info["is_holiday"],
        "holiday_name": session_info["holiday_name"],
        "is_early_close": session_info["is_early_close"],
        "early_close_reason": session_info["early_close_reason"],
        "is_market_open": session_info["is_market_open"],
        "is_regular_open": session_info["is_regular_open"],
        "positions_count": len(holdings_list),
        "total_equity": round(total_equity, 2),
        "portfolio_value": round(total_equity, 2),
        "last_equity": round(last_equity, 2),
        "daily_change_dollar": total_daily_change_dollar,
        "daily_change_pct": total_daily_change_percent,
        "total_market_value": round(total_market_value, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_unrealized_pl": round(total_unrealized_pl, 2),
        "total_unrealized_plpc": round(total_unrealized_plpc, 2),
        "total_daily_change_dollar": round(total_daily_change_dollar, 2),
        "total_daily_change_percent": round(total_daily_change_percent, 4),
        "base_prev_equity": round(last_equity, 2),
        "daily_change_source": "alpaca_last_equity",
        "open_positions_daily_dollar": round(
            sum(round(r["daily_change_dollar"], 2) for r in holdings_list), 2
        ),
        "realized_closed_today": recon["realized_closed_today"],
        "closed_today": recon["closed_today"],
        "account_daily_dollar_change": round(total_equity - last_equity, 2),
        "macro_regime": calculate_macro_regime(),
        "holdings": holdings_list,
    }

    global _LAST_VALID_PORTFOLIO_SNAPSHOT
    _LAST_VALID_PORTFOLIO_SNAPSHOT = payload
    return payload


def sanitize_holdings_for_guest(holdings_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize portfolio holdings for unauthenticated guests:
    - Obfuscate constituent symbols to LOCKED_1 .. LOCKED_N
    - Redact position-level shares, entry price, current price, weight, and factors
    - Nullify macro_regime
    - Preserve aggregate portfolio metrics (total_equity, daily_change, etc.)
    """
    raw_holdings = holdings_data.get("holdings", [])
    sanitized_holdings: List[Dict[str, Any]] = []
    for i in range(len(raw_holdings)):
        sanitized_holdings.append({
            "symbol": f"LOCKED_{i+1}",
            "qty": None,
            "avg_entry_price": None,
            "current_price": None,
            "previous_close": None,
            "daily_change_dollar": None,
            "daily_change_percent": None,
            "stock_daily_change_dollar": None,
            "price_change_percent": None,
            "opened_today": False,
            "daily_basis": None,
            "avg_fill_price_today": None,
            "market_value": None,
            "cost_basis": None,
            "unrealized_pl": None,
            "unrealized_plpc": None,
            "unrealized_plpc_raw": None,
            "side": "long",
            "weight_pct": None,
            "factors": None,
        })

    sanitized = dict(holdings_data)
    sanitized["is_guest"] = True
    sanitized["macro_regime"] = None
    sanitized["holdings"] = sanitized_holdings
    sanitized["positions_count"] = len(raw_holdings)
    return sanitized


# ── Portfolio Snapshot & Auto-Backfill Engine ─────────────────────────────────

def get_active_portfolio_holdings_and_cash() -> Tuple[Dict[str, float], float]:
    """
    Retrieve active holdings {symbol: shares} and cash balance from Alpaca,
    falling back to current_portfolio_signals.csv if Alpaca is unavailable.
    """
    holdings: Dict[str, float] = {}
    cash: float = 0.20
    try:
        client = init_trading_client(dry_run=False)
        if client:
            positions = client.get_all_positions()
            for p in positions:
                sym = str(p.symbol).strip().upper()
                qty = float(p.qty)
                if qty > 0:
                    holdings[sym] = qty
            account = client.get_account()
            if getattr(account, "cash", None) is not None:
                cash = float(account.cash)
            if holdings:
                return holdings, cash
    except Exception as exc:
        logger.warning("Could not retrieve active positions from Alpaca: %s. Checking snapshots/signals.", exc)

    # 1. Fallback to cached valid portfolio snapshot if available
    global _LAST_VALID_PORTFOLIO_SNAPSHOT
    if _LAST_VALID_PORTFOLIO_SNAPSHOT:
        raw_h = _LAST_VALID_PORTFOLIO_SNAPSHOT.get("holdings", [])
        if raw_h:
            for item in raw_h:
                sym = str(item.get("symbol", "")).strip().upper()
                qty = float(item.get("qty") or 0.0)
                if sym and qty > 0 and not sym.startswith("LOCKED_"):
                    holdings[sym] = qty
            cash = float(_LAST_VALID_PORTFOLIO_SNAPSHOT.get("cash_balance", cash))
            if holdings:
                logger.info("Using cached active portfolio holdings from snapshot (%d tickers, cash=$%0.2f)", len(holdings), cash)
                return holdings, cash

    # 2. Fallback to signals CSV (evaluate target weight with baseline price)
    if os.path.isfile(SIGNALS_CSV_PATH):
        try:
            df_sig = pd.read_csv(SIGNALS_CSV_PATH)
            last_eq = VH_BASE_EQUITY
            sig_tickers = [str(r["ticker"]).strip().upper() for _, r in df_sig.iterrows()]
            sig_prices = fetch_current_prices(sig_tickers)
            for _, row in df_sig.iterrows():
                tk = str(row["ticker"]).strip().upper()
                w = float(row.get("target_weight", 0.10))
                px = float(sig_prices.get(tk, 100.0))
                if px <= 0:
                    px = 100.0
                holdings[tk] = round((last_eq * w) / px, 4)
            logger.info("Using target holdings fallback from %s (%d tickers)", SIGNALS_CSV_PATH, len(holdings))
        except Exception as exc:
            logger.warning("Could not read signals CSV fallback: %s", exc)

    return holdings, cash


def sync_and_backfill_portfolio_history(force_refresh_all: bool = False) -> Dict[str, Any]:
    """
    Identify missing trading days between INCEPTION_DATE (or last snapshot) and latest market close.
    For each missing day, fetch historical daily close prices via yfinance for active holdings and benchmarks,
    compute NAV and returns, and persist backfilled records to SQLite and CSV so no trading sessions are skipped.
    """
    logger.info("Running portfolio history auto-backfill check...")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT date FROM portfolio_snapshots ORDER BY date ASC")
        existing_dates = set(r["date"] for r in cur.fetchall())

        # Determine official US trading days using SPY history
        today_utc = datetime.now(timezone.utc)
        today_str = today_utc.strftime("%Y-%m-%d")

        try:
            spy_data = yf.download(["SPY", "QQQ"], start=INCEPTION_DATE, progress=False, auto_adjust=True)
            if spy_data is None or spy_data.empty:
                logger.warning("Unable to fetch benchmark history for trading calendar.")
                return {"status": "skipped", "message": "Benchmark history unavailable", "backfilled_count": 0}
            close_data = spy_data["Close"] if isinstance(spy_data.columns, pd.MultiIndex) else spy_data
            canonical_days = [d.strftime("%Y-%m-%d") for d in close_data.index]
        except Exception as yf_err:
            logger.error("Error downloading benchmark calendar from yfinance: %s", yf_err)
            return {"status": "error", "message": str(yf_err), "backfilled_count": 0}

        # Completed trading days: days strictly before today, or today if after 21:00 UTC (market closed)
        completed_days = [
            d for d in canonical_days
            if (d < today_str) or (d == today_str and today_utc.hour >= 21)
        ]

        if force_refresh_all:
            missing_days = completed_days
        else:
            missing_days = [d for d in completed_days if d not in existing_dates]

        if not missing_days:
            logger.info("Portfolio history is up to date (%d sessions recorded, 0 missing).", len(completed_days))
            return {
                "status": "up_to_date",
                "backfilled_count": 0,
                "backfilled_dates": [],
                "total_records": len(existing_dates),
            }

        logger.info("Found %d missing trading days to backfill: %s", len(missing_days), missing_days)

        # Retrieve active holdings and cash
        holdings, cash = get_active_portfolio_holdings_and_cash()
        all_syms = sorted(list(set(list(holdings.keys()) + ["SPY", "QQQ"])))

        # Batch download historical daily close prices for all holdings
        try:
            px_batch = yf.download(all_syms, start=INCEPTION_DATE, progress=False, auto_adjust=True)
            px_table = px_batch["Close"] if isinstance(px_batch.columns, pd.MultiIndex) else px_batch
        except Exception as batch_err:
            logger.error("Failed to batch download historical prices for holdings: %s", batch_err)
            return {"status": "error", "message": str(batch_err), "backfilled_count": 0}

        backfilled_records = []
        for d_str in missing_days:
            matching_dt = None
            for dt_idx in px_table.index:
                if dt_idx.strftime("%Y-%m-%d") == d_str:
                    matching_dt = dt_idx
                    break
            if matching_dt is None:
                continue

            pos_val = 0.0
            for sym, qty in holdings.items():
                if sym in px_table.columns:
                    val = px_table.loc[matching_dt, sym]
                    px = float(val) if pd.notna(val) else 0.0
                    if px > 0:
                        pos_val += qty * px

            total_eq = cash + pos_val
            spy_px = float(px_table.loc[matching_dt, "SPY"]) if "SPY" in px_table.columns else SPY_INCEPTION_PRICE
            qqq_px = float(px_table.loc[matching_dt, "QQQ"]) if "QQQ" in px_table.columns else QQQ_INCEPTION_PRICE

            if d_str == INCEPTION_DATE:
                total_eq = VH_BASE_EQUITY
                cash = 0.0
                pos_val = VH_BASE_EQUITY
                spy_px = SPY_INCEPTION_PRICE
                qqq_px = QQQ_INCEPTION_PRICE
                fund_ret = 0.0
                spy_ret = 0.0
                qqq_ret = 0.0
            else:
                fund_ret = round(((total_eq / VH_BASE_EQUITY) - 1.0) * 100.0, 4)
            spy_ret = round(((spy_px / SPY_INCEPTION_PRICE) - 1.0) * 100.0, 4)
            qqq_ret = round(((qqq_px / QQQ_INCEPTION_PRICE) - 1.0) * 100.0, 4)

            up_at = f"{d_str} 16:00:00 -04:00 (Backfilled)"
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO portfolio_snapshots
                    (date, portfolio_value, cash, positions_value, spy_price, qqq_price, fund_return_pct, spy_return_pct, qqq_return_pct, updated_at, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'backfill')
                    """,
                    (
                        d_str, round(total_eq, 2), round(cash, 2), round(pos_val, 2),
                        round(spy_px, 4), round(qqq_px, 4), fund_ret, spy_ret, qqq_ret,
                        up_at,
                    ),
                )
            backfilled_records.append(d_str)
            logger.info("Backfilled trading snapshot for %s: Equity=$%0.2f (%+0.2f%%)", d_str, total_eq, fund_ret)

        # Sync to CSV
        _export_snapshots_to_csv()

        cur.execute("SELECT COUNT(*) AS cnt FROM portfolio_snapshots")
        total_count = cur.fetchone()["cnt"]

        return {
            "status": "ok",
            "backfilled_count": len(backfilled_records),
            "backfilled_dates": backfilled_records,
            "total_records": total_count,
        }
    finally:
        conn.close()


def record_daily_snapshot(force: bool = False, date_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Record an official daily snapshot of portfolio equity and benchmark levels.
    Idempotent: updates existing entry for the trade date rather than failing or duplicating.
    Saves to SQLite table `portfolio_snapshots` and syncs to `live_portfolio_history.csv`.
    """
    logger.info("Executing daily portfolio snapshot (force=%s, date_override=%s)...", force, date_override)

    # 1. Fetch live portfolio data
    portfolio_data = None
    try:
        portfolio_data = fetch_alpaca_portfolio()
    except Exception as exc:
        logger.error("Error fetching Alpaca portfolio for snapshot: %s", exc)

    trade_date = date_override
    if not trade_date and portfolio_data:
        trade_date = portfolio_data.get("trade_date")
    if not trade_date:
        trade_date = _get_eastern_now().strftime("%Y-%m-%d")

    # Weekend check
    dt = pd.to_datetime(trade_date)
    if dt.dayofweek >= 5 and not force:
        logger.info("Trade date %s falls on a weekend; skipping snapshot recording.", trade_date)
        return {
            "status": "skipped",
            "reason": "weekend",
            "trade_date": trade_date,
        }

    if portfolio_data and portfolio_data.get("total_equity") is not None:
        equity = float(portfolio_data["total_equity"])
        cash = float(portfolio_data.get("cash", 0.0))
        positions_value = float(portfolio_data.get("positions_value", equity - cash))
        timestamp_str = portfolio_data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_status = portfolio_data.get("market_status", "CLOSED")
    elif _LAST_VALID_PORTFOLIO_SNAPSHOT and _LAST_VALID_PORTFOLIO_SNAPSHOT.get("total_equity") is not None:
        # Fallback to last known live Alpaca portfolio state
        equity = float(_LAST_VALID_PORTFOLIO_SNAPSHOT["total_equity"])
        cash = float(_LAST_VALID_PORTFOLIO_SNAPSHOT.get("cash_balance", 0.0))
        positions_value = float(_LAST_VALID_PORTFOLIO_SNAPSHOT.get("positions_value", equity - cash))
        timestamp_str = _LAST_VALID_PORTFOLIO_SNAPSHOT.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_status = _LAST_VALID_PORTFOLIO_SNAPSHOT.get("market_status", "CLOSED")
    else:
        # Fallback if Alpaca is unavailable and no prior snapshot exists
        holdings, cash = get_active_portfolio_holdings_and_cash()
        prices = fetch_current_prices(list(holdings.keys()))
        positions_value = sum(holdings[s] * prices.get(s, 0.0) for s in holdings)
        equity = cash + positions_value if positions_value > 0 else VH_BASE_EQUITY
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_status = "CLOSED"

    # 2. Fetch live benchmark quotes
    benchmark_prices = {}
    for tk, fallback in (("SPY", SPY_INCEPTION_PRICE), ("QQQ", QQQ_INCEPTION_PRICE)):
        try:
            details = fetch_benchmark_extended_details(tk, market_status)
            px = details.get("price")
            benchmark_prices[tk] = float(px) if (px and px > 0) else fallback
        except Exception:
            benchmark_prices[tk] = fallback

    spy_px = benchmark_prices["SPY"]
    qqq_px = benchmark_prices["QQQ"]

    fund_return_pct = round(((equity / VH_BASE_EQUITY) - 1.0) * 100.0, 4)
    spy_return_pct = round(((spy_px / SPY_INCEPTION_PRICE) - 1.0) * 100.0, 4)
    qqq_return_pct = round(((qqq_px / QQQ_INCEPTION_PRICE) - 1.0) * 100.0, 4)

    snapshot_row = {
        "date": trade_date,
        "portfolio_value": round(equity, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "spy_price": round(spy_px, 4),
        "qqq_price": round(qqq_px, 4),
        "fund_return_pct": fund_return_pct,
        "spy_return_pct": spy_return_pct,
        "qqq_return_pct": qqq_return_pct,
        "updated_at": timestamp_str,
        "source": "live",
    }

    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_snapshots
                (date, portfolio_value, cash, positions_value, spy_price, qqq_price, fund_return_pct, spy_return_pct, qqq_return_pct, updated_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_row["date"],
                    snapshot_row["portfolio_value"],
                    snapshot_row["cash"],
                    snapshot_row["positions_value"],
                    snapshot_row["spy_price"],
                    snapshot_row["qqq_price"],
                    snapshot_row["fund_return_pct"],
                    snapshot_row["spy_return_pct"],
                    snapshot_row["qqq_return_pct"],
                    snapshot_row["updated_at"],
                    snapshot_row["source"],
                ),
            )
        logger.info("Recorded daily snapshot for %s: Equity=$%0.2f (%+0.2f%%)", trade_date, equity, fund_return_pct)
    finally:
        conn.close()

    # Sync to CSV
    _export_snapshots_to_csv()

    # Also run auto-backfill to fill any previous missing days
    backfill_res = {}
    try:
        backfill_res = sync_and_backfill_portfolio_history()
    except Exception as exc:
        logger.error("Auto-backfill during snapshot returned error: %s", exc)
        backfill_res = {"status": "error", "detail": str(exc)}

    return {
        "status": "ok",
        "snapshot": snapshot_row,
        "backfill": backfill_res,
    }


def _is_snapshot_recorded_today(today_str: str) -> bool:
    """Check if an official snapshot has already been recorded for today."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "SELECT date FROM portfolio_snapshots WHERE date = ?",
            (today_str,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


async def run_market_close_scheduler():
    """
    Lightweight async background task scheduled for Monday-Friday at market close
    (21:05 UTC / 23:05 local) to trigger the portfolio snapshot automatically without
    requiring a browser visit, plus auto-backfill on startup and after market close.
    """
    logger.info("Market close headless scheduler started (target: Mon-Fri 21:05 UTC / 23:05 local).")

    # Initial auto-backfill on server boot (brief delay so FastAPI can bind ports first)
    try:
        await asyncio.sleep(2.0)
        logger.info("Executing startup portfolio history auto-backfill check...")
        await asyncio.to_thread(sync_and_backfill_portfolio_history)
    except Exception as exc:
        logger.error("Initial background auto-backfill encountered error: %s", exc)

    last_snapshotted_day = None
    while True:
        try:
            await asyncio.sleep(30.0)
            now_utc = datetime.now(timezone.utc)
            # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4 (Skip weekends 5=Sat, 6=Sun)
            if now_utc.weekday() < 5:
                today_str = now_utc.strftime("%Y-%m-%d")
                # Trigger window: at or after 21:05 UTC on weekdays
                if (now_utc.hour > 21) or (now_utc.hour == 21 and now_utc.minute >= 5):
                    if last_snapshotted_day != today_str:
                        already_done = await asyncio.to_thread(_is_snapshot_recorded_today, today_str)
                        if not already_done:
                            logger.info(
                                "Scheduled market close trigger: recording daily snapshot for %s at %s UTC",
                                today_str,
                                now_utc.strftime("%H:%M:%S"),
                            )
                            await asyncio.to_thread(record_daily_snapshot)
                        last_snapshotted_day = today_str
        except asyncio.CancelledError:
            logger.info("Market close headless scheduler cancelled.")
            break
        except Exception as loop_err:
            logger.exception("Error in market close background scheduler loop: %s", loop_err)
            await asyncio.sleep(60.0)


def verify_snapshot_auth(request: Request, token: Optional[str] = None) -> bool:
    """
    Verify caller is authorized to trigger snapshots/backfill.
    Permitted if:
    1. Header X-Internal-Token or query param 'token' matches INTERNAL_CRON_TOKEN or TOKEN_SECRET.
    2. Bearer token in Authorization header matches INTERNAL_CRON_TOKEN or TOKEN_SECRET.
    3. Caller provides a valid session token belonging to an 'admin' role user.
    """
    internal_keys = {TOKEN_SECRET, INTERNAL_CRON_TOKEN}

    # 1. Check direct query token
    if token and token.strip() in internal_keys:
        return True

    # 2. Check X-Internal-Token header
    header_token = request.headers.get("X-Internal-Token")
    if header_token and header_token.strip() in internal_keys:
        return True

    # 3. Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        bearer_val = auth_header[7:].strip()
        if bearer_val in internal_keys:
            return True
        user = verify_auth_token(bearer_val)
        if user and user.get("role") == "admin":
            return True

    # 4. Check query token against admin session token
    if token:
        user = verify_auth_token(token.strip())
        if user and user.get("role") == "admin":
            return True

    return False


# ── FastAPI Application Lifespan & CORS Setup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown hooks."""
    init_db()
    load_backtest_cache()
    scheduler_task = asyncio.create_task(run_market_close_scheduler())
    yield
    scheduler_task.cancel()
    try:
        await scheduler_task
    except (asyncio.CancelledError, Exception):
        pass


app = FastAPI(
    title="The Value Hunter Telemetry & Investor Portal API",
    description="Live portfolio, benchmark, authentication, holdings and automated snapshot backend for The Value Hunter fund.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Snapshot & History Endpoints ──────────────────────────────────────────────

@app.post("/api/portfolio/snapshot")
def trigger_portfolio_snapshot(
    request: Request,
    token: Optional[str] = Query(default=None),
    force: bool = Query(default=False),
    date_override: Optional[str] = Query(default=None),
):
    """
    Idempotent endpoint to record the official daily portfolio snapshot.
    Secured via internal cron token or admin authentication.
    """
    if not verify_snapshot_auth(request, token):
        return JSONResponse(
            status_code=403,
            content={"status": "error", "message": "Unauthorized. Internal cron token or admin authorization required."},
        )

    try:
        result = record_daily_snapshot(force=force, date_override=date_override)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        logger.exception("Error executing /api/portfolio/snapshot: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/api/portfolio/backfill")
def trigger_portfolio_backfill(
    request: Request,
    token: Optional[str] = Query(default=None),
    force_all: bool = Query(default=False),
):
    """
    Idempotent endpoint to detect and backfill missing historical trading sessions.
    Secured via internal cron token or admin authentication.
    """
    if not verify_snapshot_auth(request, token):
        return JSONResponse(
            status_code=403,
            content={"status": "error", "message": "Unauthorized. Internal cron token or admin authorization required."},
        )

    try:
        result = sync_and_backfill_portfolio_history(force_refresh_all=force_all)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        logger.exception("Error executing /api/portfolio/backfill: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.get("/api/portfolio/history")
@app.get("/api/chart-history")
def get_portfolio_history(
    request: Request,
    auto_backfill: bool = Query(default=False),
    token: Optional[str] = Query(default=None),
):
    """
    Retrieve full chronological daily portfolio history and benchmark performance records.
    Public endpoint: serves aggregate normalized equity and benchmark curves (zero ticker data).
    """
    try:
        caller = extract_request_user(request, token)
        if auto_backfill and caller and (caller.get("status") == "approved" or caller.get("role") == "admin"):
            sync_and_backfill_portfolio_history()

        conn = get_db_connection()
        try:
            cur = conn.execute(
                """
                SELECT date, portfolio_value, cash, positions_value, spy_price, qqq_price,
                       fund_return_pct, spy_return_pct, qqq_return_pct, updated_at, source
                FROM portfolio_snapshots
                WHERE strftime('%w', date) NOT IN ('0', '6') AND date != '2026-11-01'
                ORDER BY date ASC
                """
            )
            raw_rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        # Strict filter: keep strictly Monday through Friday trading days
        rows = []
        for r in raw_rows:
            try:
                dt_obj = datetime.strptime(str(r["date"]).strip(), "%Y-%m-%d")
                if dt_obj.weekday() < 5:
                    rows.append(r)
            except Exception:
                rows.append(r)

        for r in rows:
            if r.get("date") == INCEPTION_DATE:
                r["portfolio_value"] = VH_BASE_EQUITY
                r["fund_return_pct"] = 0.0
                r["spy_return_pct"] = 0.0
                r["qqq_return_pct"] = 0.0
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "count": len(rows),
                "inception_date": INCEPTION_DATE,
                "base_equity": VH_BASE_EQUITY,
                "history": rows,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/portfolio/history: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


# ── Historical Backtest Endpoint (2005-2025) ──────────────────────────────────

@app.get("/api/portfolio/backtest")
def get_portfolio_backtest(
    resolution: str = Query(default="monthly", pattern="^(monthly|daily|yearly)$"),
    refresh: bool = Query(default=False),
):
    """
    Return comprehensive historical backtest results (2005-2025).
    Includes summary metrics (CAGR, Vol, Sharpe, Max Drawdown, Calmar),
    annual calendar returns comparison vs SPY, and interactive series curves
    (Equity curve & Underwater drawdowns) in 'daily', 'monthly', or 'yearly' resolution.
    Serves instantly from precomputed in-memory cache.
    """
    data = load_backtest_cache(force_reload=refresh)
    if not data:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Backtest results not yet available."},
        )

    series_data = data.get(f"{resolution}_series")
    if not series_data:
        if resolution == "yearly" and "monthly_series" in data:
            ms = data["monthly_series"]
            qqq_eq = ms.get("qqq_equity", [])
            qqq_dd = ms.get("qqq_drawdown", [])
            year_pts = {}
            for i, (d, s, sp, s_dd, sp_dd) in enumerate(zip(
                ms["dates"], ms["strategy_equity"], ms["spy_equity"], ms["strategy_drawdown"], ms["spy_drawdown"]
            )):
                q_val = qqq_eq[i] if i < len(qqq_eq) else 0
                q_dd = qqq_dd[i] if i < len(qqq_dd) else 0
                year_pts[d[:4]] = (d, s, sp, q_val, s_dd, sp_dd, q_dd)
            series_data = {
                "dates": [v[0] for v in year_pts.values()],
                "strategy_equity": [v[1] for v in year_pts.values()],
                "spy_equity": [v[2] for v in year_pts.values()],
                "qqq_equity": [v[3] for v in year_pts.values()],
                "strategy_drawdown": [v[4] for v in year_pts.values()],
                "spy_drawdown": [v[5] for v in year_pts.values()],
                "qqq_drawdown": [v[6] for v in year_pts.values()],
            }
        else:
            series_data = data.get("monthly_series")

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "generated_at": data.get("generated_at"),
            "period": data.get("period"),
            "strategy_specs": data.get("strategy_specs"),
            "summary_metrics": data.get("summary_metrics"),
            "annual_returns": data.get("annual_returns"),
            "resolution": resolution,
            "series": series_data,
            "monthly_series": data.get("monthly_series"),
        },
    )


# ── Dashboard Root Endpoint ───────────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    """Serve `index.html` directly at http://localhost:8000/."""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "index.html not found in project root."},
    )


@app.get("/healthz")
def health_check():
    """Lightweight liveness probe for cloud platforms (Render, Fly.io, Cloud Run)."""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ── Authentication Endpoints ──────────────────────────────────────────────────

@app.post("/api/auth/register")
def register_user(payload: RegisterRequest):
    """Register a new investor account with status='pending'."""
    try:
        full_name = payload.full_name.strip()
        email = payload.email.strip().lower()
        phone = payload.phone.strip()
        password = payload.password

        if not full_name or not email or not phone or not password:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "All fields (Full Name, Email, Phone, Password) are required."},
            )

        if len(password) < 6:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Password must be at least 6 characters long."},
            )

        existing = get_user_by_email(email)
        if existing:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "An account with this email address is already registered."},
            )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pw_hash = hash_password(password)

        conn = get_db_connection()
        try:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO users (full_name, email, phone, password_hash, role, status, created_at)
                    VALUES (?, ?, ?, ?, 'user', 'pending', ?)
                    """,
                    (full_name, email, phone, pw_hash, now_str),
                )
                new_id = cur.lastrowid
        finally:
            conn.close()

        user_info = get_user_by_id(int(new_id))
        logger.info("New user registered (pending approval): %s (%s)", full_name, email)

        return JSONResponse(
            status_code=201,
            content={
                "status": "ok",
                "message": "Registration submitted successfully. Your account is pending admin approval.",
                "user": user_info,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/auth/register: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Registration failed: {exc}"},
        )


@app.post("/api/auth/login")
def login_user(payload: LoginRequest):
    """Validate user credentials and return user profile, role, approval status, and session token."""
    try:
        email = payload.email.strip().lower()
        password = payload.password

        user_record = get_user_by_email(email)
        if not user_record and email in ADMIN_EMAILS_SET:
            init_db()
            user_record = get_user_by_email(email)

        # Authenticate solely against the stored PBKDF2 hash (no hardcoded bypasses).
        if not user_record or not verify_password(password, user_record["password_hash"]):
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Invalid email or password."},
            )

        user_clean = {k: v for k, v in user_record.items() if k != "password_hash"}
        if email in ADMIN_EMAILS_SET :
            user_clean["role"] = "admin"
            user_clean["status"] = "approved"

        # Check approval status
        if user_clean["status"] == "pending":
            return JSONResponse(
                status_code=403,
                content={
                    "status": "pending",
                    "message": "Your account is pending admin approval. You will be notified once approved.",
                    "user": user_clean,
                },
            )

        if user_clean["status"] == "rejected":
            return JSONResponse(
                status_code=403,
                content={
                    "status": "rejected",
                    "message": "Your account registration request was declined by the administrator.",
                    "user": user_clean,
                },
            )

        token = create_auth_token(user_clean)
        logger.info("User logged in: %s (role=%s, status=%s)", user_clean["email"], user_clean["role"], user_clean["status"])

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Login successful.",
                "token": token,
                "user": user_clean,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/auth/login: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Login error: {exc}"},
        )


OTP_EXPIRY_MINUTES = 10
OTP_MAX_ATTEMPTS = 3


@app.post("/api/auth/request-otp")
def request_email_otp(payload: RequestOtpPayload):
    """
    Generate a 6-digit Email OTP code (10-minute expiration) for an existing,
    approved investor or admin account, store its hash in `email_otps`, and return
    the EmailJS dispatch payload.
    """
    try:
        email = payload.email.strip().lower()
        if not email:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Email address is required."},
            )

        user_record = get_user_by_email(email)
        if not user_record and email in ADMIN_EMAILS_SET:
            init_db()
            user_record = get_user_by_email(email)

        if not user_record:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No registered investor account found for this email address.",
                },
            )

        user_clean = {k: v for k, v in user_record.items() if k != "password_hash"}
        if email in ADMIN_EMAILS_SET :
            user_clean["role"] = "admin"
            user_clean["status"] = "approved"

        # Verify user is approved or admin
        if user_clean["role"] != "admin":
            if user_clean["status"] == "pending":
                return JSONResponse(
                    status_code=403,
                    content={
                        "status": "pending",
                        "message": "Your account is pending admin approval. You cannot request a login code until approved.",
                    },
                )
            if user_clean["status"] == "rejected":
                return JSONResponse(
                    status_code=403,
                    content={
                        "status": "rejected",
                        "message": "Your account registration request was declined by the administrator.",
                    },
                )
            if user_clean["status"] != "approved":
                return JSONResponse(
                    status_code=403,
                    content={
                        "status": "error",
                        "message": "Your investor account is not approved for portal access.",
                    },
                )

        # Generate secure random 6-digit code
        otp_code = f"{secrets.randbelow(1_000_000):06d}"
        code_hash = hash_password(otp_code)
        expires_dt = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
        expires_at_str = expires_dt.strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    "DELETE FROM email_otps WHERE lower(trim(email)) = lower(trim(?))",
                    (email,),
                )
                conn.execute(
                    """
                    INSERT INTO email_otps (email, code_hash, expires_at, attempts)
                    VALUES (?, ?, ?, 0)
                    """,
                    (email, code_hash, expires_at_str),
                )
        finally:
            conn.close()

        logger.info(
            "Generated 6-digit OTP for %s (expires at %s) [OTP: %s]",
            email,
            expires_at_str,
            otp_code,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": f"Verification code sent to {email}. Valid for {OTP_EXPIRY_MINUTES} minutes.",
                "email": email,
                "otp_code": otp_code,
                "expires_in_seconds": OTP_EXPIRY_MINUTES * 60,
                "expires_at": expires_at_str,
                "emailjs_payload": {
                    "to_email": email,
                    "otp_code": otp_code,
                },
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/auth/request-otp: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to generate verification code: {exc}"},
        )


@app.post("/api/auth/verify-otp")
def verify_email_otp(payload: VerifyOtpPayload):
    """
    Validate a 6-digit Email OTP code, enforcing 10-minute expiration and max 3 wrong attempts.
    On success, invalidates the OTP and returns the authenticated user session profile and bearer token.
    """
    try:
        email = payload.email.strip().lower()
        code = payload.code.strip()

        if not email or not code:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Email and 6-digit verification code are required."},
            )

        conn = get_db_connection()
        try:
            cur = conn.execute(
                """
                SELECT email, code_hash, expires_at, attempts
                FROM email_otps
                WHERE lower(email) = lower(?)
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (email,),
            )
            otp_row = cur.fetchone()

            if not otp_row:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "No active verification code found for this email. Please request a new code.",
                    },
                )

            attempts = int(otp_row["attempts"] or 0)
            if attempts >= OTP_MAX_ATTEMPTS:
                with conn:
                    conn.execute("DELETE FROM email_otps WHERE lower(email) = lower(?)", (email,))
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "message": "Maximum verification attempts (3) exceeded. Please request a new code.",
                    },
                )

            # Check expiration timestamp
            expires_at_raw = str(otp_row["expires_at"])
            try:
                expires_dt = datetime.strptime(expires_at_raw, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                expires_dt = datetime.fromisoformat(expires_at_raw)

            if datetime.now() > expires_dt:
                with conn:
                    conn.execute("DELETE FROM email_otps WHERE lower(email) = lower(?)", (email,))
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Verification code has expired. Please request a new 6-digit code.",
                    },
                )

            # Validate code hash
            if not verify_password(code, otp_row["code_hash"]):
                new_attempts = attempts + 1
                if new_attempts >= OTP_MAX_ATTEMPTS:
                    with conn:
                        conn.execute("DELETE FROM email_otps WHERE lower(email) = lower(?)", (email,))
                    return JSONResponse(
                        status_code=429,
                        content={
                            "status": "error",
                            "message": "Invalid verification code. Maximum attempts (3/3) reached — please request a new code.",
                            "attempts_remaining": 0,
                        },
                    )
                else:
                    remaining = OTP_MAX_ATTEMPTS - new_attempts
                    with conn:
                        conn.execute(
                            "UPDATE email_otps SET attempts = ? WHERE lower(email) = lower(?)",
                            (new_attempts, email),
                        )
                    return JSONResponse(
                        status_code=401,
                        content={
                            "status": "error",
                            "message": f"Invalid verification code. {remaining} attempt(s) remaining.",
                            "attempts_remaining": remaining,
                        },
                    )

            # Valid code -> invalidate OTP immediately
            with conn:
                conn.execute("DELETE FROM email_otps WHERE lower(email) = lower(?)", (email,))
        finally:
            conn.close()

        user_record = get_user_by_email(email)
        if not user_record and email in ADMIN_EMAILS_SET:
            init_db()
            user_record = get_user_by_email(email)

        if not user_record:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "User account no longer exists."},
            )

        user_clean = {k: v for k, v in user_record.items() if k != "password_hash"}
        if email in ADMIN_EMAILS_SET :
            user_clean["role"] = "admin"
            user_clean["status"] = "approved"
        if user_clean["role"] != "admin" and user_clean["status"] != "approved":
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Account is not approved for access."},
            )

        token = create_auth_token(user_clean)
        logger.info("User logged in via Email OTP: %s (role=%s)", user_clean["email"], user_clean["role"])

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "OTP login successful.",
                "token": token,
                "user": user_clean,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/auth/verify-otp: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"OTP verification failed: {exc}"},
        )


# ── Admin Management Endpoints ────────────────────────────────────────────────

@app.get("/api/admin/users")
def list_users_for_admin(
    request: Request,
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Query(default=None),
):
    """Protected endpoint (admin only): list all pending and registered users."""
    try:
        caller = extract_request_user(request, token or auth_token)
        if not caller or str(caller.get("role", "")).strip().lower() != "admin":
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Admin privileges required."},
            )

        conn = get_db_connection()
        try:
            cur = conn.execute(
                """
                SELECT id, full_name, email, phone, role, status, created_at
                FROM users
                ORDER BY
                    CASE lower(status)
                        WHEN 'pending' THEN 0
                        WHEN 'approved' THEN 1
                        ELSE 2
                    END,
                    CASE WHEN lower(role) = 'admin' THEN 0 ELSE 1 END,
                    created_at DESC
                """
            )
            raw_users = [_row_to_user_dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        # Deduplicate users by lower(email), strictly prioritizing ADMIN and filtering out redundant USER rows
        seen_emails = set()
        users = []
        admin_email_lower = DEFAULT_ADMIN_EMAIL.lower().strip()
        for u in raw_users:
            em = str(u.get("email") or "").strip().lower()
            if not em:
                continue
            # Remove redundant non-admin duplicate for the admin email
            if em == admin_email_lower and str(u.get("role") or "").strip().lower() != "admin":
                continue
            if em in seen_emails:
                continue
            seen_emails.add(em)
            users.append(u)

        # Ensure ordered by pending first, then approved, then others
        def _sort_key(u):
            st = str(u.get("status") or "").strip().lower()
            ro = str(u.get("role") or "").strip().lower()
            status_prio = 0 if st == "pending" else (1 if st == "approved" else 2)
            role_prio = 0 if ro == "admin" else 1
            return (status_prio, role_prio)

        users.sort(key=_sort_key)

        pending_count = sum(1 for u in users if str(u.get("status") or "").strip().lower() == "pending")
        approved_count = sum(1 for u in users if u["status"] == "approved")
        rejected_count = sum(1 for u in users if u["status"] == "rejected")

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "total": len(users),
                "pending_count": pending_count,
                "approved_count": approved_count,
                "rejected_count": rejected_count,
                "users": users,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/admin/users: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to fetch users: {exc}"},
        )


@app.post("/api/admin/update-status")
def admin_update_user_status(
    payload: UpdateStatusRequest,
    request: Request,
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Query(default=None),
):
    """Protected endpoint (admin only): update a user's status to 'approved' or 'rejected'."""
    try:
        caller = extract_request_user(request, token or auth_token)
        if not caller or str(caller.get("role", "")).strip().lower() != "admin":
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Admin privileges required."},
            )

        new_status = payload.status.strip().lower()
        if new_status not in ("approved", "rejected", "pending"):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid status. Must be 'approved' or 'rejected'."},
            )

        conn = get_db_connection()
        target_user = None
        try:
            with conn:
                if payload.user_id is not None:
                    conn.execute(
                        "UPDATE users SET status = ? WHERE id = ?",
                        (new_status, int(payload.user_id)),
                    )
                    cur = conn.execute("SELECT * FROM users WHERE id = ?", (int(payload.user_id),))
                    row = cur.fetchone()
                    if row:
                        target_user = _row_to_user_dict(row)
                elif payload.email:
                    conn.execute(
                        "UPDATE users SET status = ? WHERE lower(email) = lower(?)",
                        (new_status, payload.email.strip()),
                    )
                    cur = conn.execute(
                        "SELECT * FROM users WHERE lower(email) = lower(?)",
                        (payload.email.strip(),),
                    )
                    row = cur.fetchone()
                    if row:
                        target_user = _row_to_user_dict(row)
        finally:
            conn.close()

        if not target_user:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Target user not found."},
            )

        logger.info(
            "Admin %s updated user %s (%s) status -> %s",
            caller["email"],
            target_user["full_name"],
            target_user["email"],
            new_status,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": f"User {target_user['full_name']} ({target_user['email']}) marked as {new_status}.",
                "user": target_user,
            },
        )
    except Exception as exc:
        logger.exception("Error in /api/admin/update-status: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to update status: {exc}"},
        )


# ── Protected Investor Holdings Endpoint ──────────────────────────────────────

@app.get("/api/portfolio/holdings")
def get_portfolio_holdings(
    request: Request,
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Query(default=None),
):
    """
    Returns portfolio holdings breakdown.
    For approved investors & admin: returns real constituent positions and factors.
    For unauthenticated guests / unapproved users: returns sanitized holdings (LOCKED_1..10, no factors).
    """
    try:
        caller = extract_request_user(request, token or auth_token)
        is_approved = bool(caller and (caller.get("status") == "approved" or caller.get("role") == "admin"))

        try:
            holdings_data = fetch_alpaca_holdings()
            if not is_approved:
                holdings_data = sanitize_holdings_for_guest(holdings_data)
            else:
                holdings_data["is_guest"] = False

            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **holdings_data,
                },
            )
        except Exception as alpaca_err:
            logger.error("Failed to fetch Alpaca holdings: %s", alpaca_err)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Unable to retrieve live holdings from Alpaca API.",
                    "detail": str(alpaca_err),
                    "holdings": [],
                },
            )
    except Exception as exc:
        logger.exception("Unexpected error in /api/portfolio/holdings: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {exc}"},
        )


@app.get("/api/stream/holdings")
async def stream_portfolio_holdings(
    request: Request,
    token: Optional[str] = Query(default=None),
    auth_token: Optional[str] = Query(default=None),
):
    """
    Real-time Server-Sent Events (SSE) stream for active holdings telemetry.
    Streams live prices, 1D change, market value, and unrealized P&L every 1.5 - 2s.
    For approved investors: streams full constituent symbols and macro regime.
    For guests: streams sanitized holdings and macro_regime: null.
    """
    caller = extract_request_user(request, token or auth_token)
    is_approved = bool(caller and (caller.get("status") == "approved" or caller.get("role") == "admin"))
    user_label = caller.get("email") if caller else "guest"

    async def event_generator():
        logger.info("SSE client connected to /api/stream/holdings (%s)", user_label)
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected from /api/stream/holdings (%s)", user_label)
                    break

                try:
                    holdings_data = await asyncio.to_thread(fetch_alpaca_holdings)
                except Exception as alpaca_exc:
                    logger.warning("Error fetching holdings in SSE stream: %s", alpaca_exc)
                    if _LAST_VALID_PORTFOLIO_SNAPSHOT is not None:
                        # Retain latest valid snapshot during overnight / holiday / broker downtime
                        holdings_data = dict(_LAST_VALID_PORTFOLIO_SNAPSHOT)
                        s_info = resolve_market_session(None)
                        holdings_data.update(s_info)
                    else:
                        yield f": keepalive\n\n"
                        await asyncio.sleep(2.0)
                        continue

                market_status = holdings_data.get("market_status", "REGULAR")
                session_phase = holdings_data.get("session_phase") or holdings_data.get("market_phase", "REGULAR")
                holdings_data["session_phase"] = session_phase

                if is_approved:
                    live_spy_price = await asyncio.to_thread(fetch_live_spy_price, market_status)
                    macro_payload = calculate_macro_regime(live_spy_price)
                    holdings_data["macro_regime"] = macro_payload
                    holdings_data["is_guest"] = False
                else:
                    holdings_data = sanitize_holdings_for_guest(holdings_data)

                # Ensure session fields are in root of payload
                holdings_data["session_phase"] = session_phase

                payload = json.dumps({
                    "status": "ok",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **holdings_data,
                })
                yield f"data: {payload}\n\n"

                if request.query_params.get("once") == "1" or request.headers.get("X-Test-Single-Chunk") == "1":
                    break

                await asyncio.sleep(1.75)
        except asyncio.CancelledError:
            logger.info("SSE holdings stream cancelled for client %s", user_label)
        except Exception as exc:
            logger.info("SSE holdings stream closed: %s", exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )


# ── Protected Live Telemetry Endpoint ─────────────────────────────────────────

@app.get("/api/live-stats")
@app.get("/api/portfolio/live")
def get_live_stats(request: Request, token: Optional[str] = Query(default=None)):
    """
    Return live Alpaca portfolio telemetry and real-time SPY/QQQ benchmark quotes.
    Publicly accessible: returns aggregate equity and benchmarks.
    Guests receive is_guest=True and macro_regime=None.
    Approved investors receive macro_regime and full portfolio data.
    """
    try:
        caller = extract_request_user(request, token)
        is_approved = bool(caller and (caller.get("status") == "approved" or caller.get("role") == "admin"))
        is_guest = not is_approved

        errors = {}

        # 1. Fetch Live Alpaca Portfolio Data (includes extended/overnight hours recalculation)
        portfolio_data = None
        session_info = resolve_market_session(None)
        market_status = session_info["market_status"]
        market_phase = session_info["market_phase"]
        session_phase = session_info["session_phase"]
        session_name = session_info["session_name"]
        is_holiday = session_info["is_holiday"]
        holiday_name = session_info["holiday_name"]
        is_early_close = session_info["is_early_close"]
        is_market_open = session_info["is_market_open"]
        is_regular_open = session_info["is_regular_open"]

        try:
            portfolio_data = fetch_alpaca_portfolio()
            market_status = portfolio_data.get("market_status", market_status)
            market_phase = portfolio_data.get("market_phase", market_phase)
            session_phase = portfolio_data.get("session_phase", session_phase)
            session_name = portfolio_data.get("session_name", session_name)
            is_holiday = portfolio_data.get("is_holiday", is_holiday)
            holiday_name = portfolio_data.get("holiday_name", holiday_name)
            is_early_close = portfolio_data.get("is_early_close", is_early_close)
            is_market_open = portfolio_data.get("is_market_open", is_market_open)
            is_regular_open = portfolio_data.get("is_regular_open", is_regular_open)
        except Exception as alpaca_err:
            logger.error("Failed to fetch Alpaca portfolio data: %s", alpaca_err)
            errors["alpaca"] = str(alpaca_err)

        # 2. Fetch Live Benchmark Quotes & 1D Change (SPY & QQQ) with Pre-Market / After-Hours / Overnight support
        benchmark_data: Dict[str, Any] = {}
        benchmark_details: Dict[str, Any] = {}
        base_prices = {"SPY": SPY_INCEPTION_PRICE, "QQQ": QQQ_INCEPTION_PRICE}
        for ticker in ("SPY", "QQQ"):
            try:
                details = fetch_benchmark_extended_details(ticker, market_status)
                quote_px = float(details["price"])
                base_px = base_prices.get(ticker, quote_px)
                sim_val = VH_BASE_EQUITY * (quote_px / base_px) if base_px > 0 else VH_BASE_EQUITY
                sim_day_dollar = round(sim_val * (float(details["daily_return_pct"]) / 100.0), 2)
                details["quote_daily_dollar_change"] = details["daily_dollar_change"]
                details["daily_dollar_change"] = sim_day_dollar
                details["simulated_value"] = round(sim_val, 2)
                benchmark_data[ticker] = details["price"]
                benchmark_data[f"{ticker}_previous_close"] = details["previous_close"]
                benchmark_data[f"{ticker}_daily_return_pct"] = details["daily_return_pct"]
                benchmark_data[f"{ticker}_daily_dollar_change"] = sim_day_dollar
                benchmark_details[ticker] = details
            except Exception as yf_err:
                logger.error("Failed to fetch live quote for %s: %s", ticker, yf_err)
                benchmark_data[ticker] = None
                benchmark_details[ticker] = {
                    "price": None,
                    "previous_close": None,
                    "daily_return_pct": None,
                    "daily_dollar_change": None,
                }
                errors[f"yfinance_{ticker}"] = str(yf_err)

        benchmark_data["details"] = benchmark_details

        # Build response payload
        response_payload: Dict[str, Any] = {
            "status": "ok" if not errors else "partial_error",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_status": market_status,
            "session_name": session_name,
            "market_phase": market_phase,
            "session_phase": session_phase,
            "is_market_open": is_market_open,
            "is_regular_open": is_regular_open,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "is_early_close": is_early_close,
            "inception_date": INCEPTION_DATE,
            "initial_capital": VH_BASE_EQUITY,
            "vh_base_equity": VH_BASE_EQUITY,
            "is_guest": is_guest,
            "portfolio": portfolio_data or {
                "total_equity": None,
                "equity": None,
                "last_equity": None,
                "daily_dollar_change": None,
                "daily_return_pct": None,
                "cash_balance": None,
                "cash": None,
                "buying_power": None,
                "market_status": market_status,
                "session_name": session_name,
                "market_phase": market_phase,
                "session_phase": session_phase,
                "is_market_open": is_market_open,
                "is_regular_open": is_regular_open,
                "is_holiday": is_holiday,
                "holiday_name": holiday_name,
                "is_early_close": is_early_close,
            },
            "benchmarks": benchmark_data,
            "macro_regime": calculate_macro_regime() if is_approved else None,
        }

        # Convenience top-level fields
        if portfolio_data:
            response_payload["total_equity"] = portfolio_data["total_equity"]
            response_payload["portfolio_value"] = portfolio_data["total_equity"]
            response_payload["last_equity"] = portfolio_data["last_equity"]
            response_payload["daily_change_dollar"] = portfolio_data["daily_change_dollar"]
            response_payload["daily_change_pct"] = portfolio_data["daily_change_pct"]
            response_payload["daily_dollar_change"] = portfolio_data["daily_dollar_change"]
            response_payload["daily_return_pct"] = portfolio_data["daily_return_pct"]
            response_payload["cash_balance"] = portfolio_data["cash_balance"]
            response_payload["buying_power"] = portfolio_data["buying_power"]

        if errors:
            response_payload["errors"] = errors

        status_code = 200 if (portfolio_data or any(benchmark_data.values())) else 503
        return JSONResponse(status_code=status_code, content=response_payload)

    except Exception as exc:
        logger.exception("Unexpected error in /api/live-stats: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error while fetching live stats.",
                "detail": str(exc),
            },
        )


# ── Protected Portfolio Signals & Orders Endpoints ────────────────────────────

@app.get("/api/portfolio/macro-regime")
def get_macro_regime_route(request: Request, token: Optional[str] = Query(default=None)):
    """Returns current Macro Regime indicator, 10M SMA distance, and target allocation. Protected."""
    caller = extract_request_user(request, token)
    if not caller or (caller.get("status") != "approved" and caller.get("role") != "admin"):
        return JSONResponse(
            status_code=403,
            content={"status": "error", "message": "Investor approval required to view macro regime."},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "macro_regime": get_current_macro_regime()},
    )


@app.get("/api/portfolio/signals")
@app.get("/api/portfolio/rebalance-signals")
def get_portfolio_signals(request: Request, token: Optional[str] = Query(default=None)):
    """
    Protected endpoint: Returns latest multi-factor rebalance signals.
    Requires an authenticated user session.
    """
    caller = extract_request_user(request, token)
    if not caller:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Authentication required to view rebalance signals."},
        )

    macro_regime = get_current_macro_regime()
    if os.path.exists(SIGNALS_JSON_PATH):
        try:
            with open(SIGNALS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "ok",
                        "count": data.get("count", len(data.get("signals", []))),
                        "macro_regime": data.get("macro_regime", macro_regime),
                        "signals": data.get("signals", []),
                    },
                )
        except Exception as e:
            logger.warning("Error reading %s: %s", SIGNALS_JSON_PATH, e)
    signals_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_portfolio_signals.csv")
    if os.path.exists(signals_file):
        try:
            df = pd.read_csv(signals_file)
            records = json.loads(df.to_json(orient="records"))
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "count": len(records),
                    "macro_regime": macro_regime,
                    "signals": records,
                },
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "current_portfolio_signals.csv not found."},
    )


@app.get("/api/portfolio/orders")
@app.get("/api/portfolio/live-orders")
def get_portfolio_orders(request: Request, token: Optional[str] = Query(default=None)):
    """
    Protected endpoint: Returns execution order log and trade history.
    Requires an authenticated user session.
    """
    caller = extract_request_user(request, token)
    if not caller:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Authentication required to view portfolio orders."},
        )

    macro_regime = get_current_macro_regime()
    orders_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpaca_trade_log.csv")
    if os.path.exists(orders_file):
        try:
            df = pd.read_csv(orders_file)
            records = json.loads(df.to_json(orient="records"))
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "count": len(records),
                    "macro_regime": macro_regime,
                    "orders": records,
                },
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "alpaca_trade_log.csv not found."},
    )


if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
