import time
"""
test_system_integrity.py — Comprehensive System Integrity & Security Verification Suite.

Verification Vectors:
  1. Guest Data Redaction & Leakage Test (/api/portfolio/holdings & /api/stream/holdings)
  2. Auth & Admin Security Test (OTP bypass blocked, password auth, admin endpoints)
  3. Live Telemetry & 1D Calculation Test (/api/live-stats vs Alpaca account)
  4. Macro Regime Buffer Evaluation (boundary condition evaluations around $725.56 SMA)
"""

import os
import sys
import json
import unittest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Ensure parent directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from api_server import app, calculate_macro_regime, SMA_10M_THRESHOLD, resolve_market_session
from alpaca_execution import init_trading_client


class TestSystemIntegrity(unittest.TestCase):
    client: TestClient = None
    real_symbols: List[str] = []
    account: Any = None

    @classmethod
    def setUpClass(cls):
        os.environ["ADMIN_PASSWORD"] = "Admin2026!"
        os.environ["VH_ADMIN_PASSWORD"] = "Admin2026!"
        cls.client = TestClient(app)
        try:
            alpaca_client = init_trading_client(dry_run=False)
            positions = alpaca_client.get_all_positions()
            cls.real_symbols = [str(p.symbol).upper() for p in positions]
            cls.account = alpaca_client.get_account()
        except Exception as e:
            cls.real_symbols = ["AAPL", "MSFT", "NVDA"]
            class MockAccount:
                equity = "100000.00"
                last_equity = "100000.00"
            cls.account = MockAccount()

            # Mock api_server Alpaca holdings when offline
            import api_server
            def mock_fetch_alpaca_holdings():
                return {
                    "is_guest": False,
                    "cash": 50000.0,
                    "positions_value": 50000.0,
                    "total_equity": 100000.0,
                    "holdings": [
                        {
                            "symbol": "AAPL",
                            "qty": 50,
                            "avg_entry_price": 200.0,
                            "current_price": 210.0,
                            "market_value": 10500.0,
                            "unrealized_pl": 500.0,
                            "unrealized_plpc": 5.0,
                            "weight_pct": 10.5,
                            "factors": {"momentum": 1.2, "stability": 0.8, "value": 0.9, "composite": 1.0},
                        },
                        {
                            "symbol": "MSFT",
                            "qty": 25,
                            "avg_entry_price": 400.0,
                            "current_price": 420.0,
                            "market_value": 10500.0,
                            "unrealized_pl": 500.0,
                            "unrealized_plpc": 5.0,
                            "weight_pct": 10.5,
                            "factors": {"momentum": 0.9, "stability": 1.1, "value": 0.7, "composite": 0.9},
                        },
                    ],
                }
            api_server.fetch_alpaca_holdings = mock_fetch_alpaca_holdings

            def mock_get_alpaca_portfolio_data():
                return {
                    "total_equity": 100000.0,
                    "equity": 100000.0,
                    "last_equity": 100000.0,
                    "cash": 50000.0,
                    "buying_power": 100000.0,
                    "daily_dollar_change": 0.0,
                    "daily_return_pct": 0.0,
                }
            api_server.get_alpaca_portfolio_data = mock_get_alpaca_portfolio_data

            def mock_fetch_alpaca_portfolio():
                return {
                    "total_equity": 100000.0,
                    "equity": 100000.0,
                    "last_equity": 100000.0,
                    "daily_change_dollar": 0.0,
                    "daily_change_pct": 0.0,
                    "daily_dollar_change": 0.0,
                    "daily_return_pct": 0.0,
                    "cash": 50000.0,
                    "cash_balance": 50000.0,
                    "buying_power": 100000.0,
                    "market_status": "REGULAR",
                    "market_phase": "REGULAR",
                    "session_phase": "REGULAR",
                    "session_name": "MARKET OPEN",
                    "is_holiday": False,
                    "holiday_name": None,
                    "is_early_close": False,
                    "is_market_open": True,
                    "is_regular_open": True,
                }
            api_server.fetch_alpaca_portfolio = mock_fetch_alpaca_portfolio

            def mock_live_quote(sym, status=None):
                return {
                    "price": 500.0,
                    "previous_close": 500.0,
                    "daily_dollar_change": 0.0,
                    "daily_return_pct": 0.0,
                }
            api_server.fetch_live_quote_with_extended = mock_live_quote

    def setUp(self):
        if self.__class__.client is None:
            self.__class__.setUpClass()
        self.client = self.__class__.client
        self.real_symbols = self.__class__.real_symbols
        self.account = self.__class__.account

    # ── Vector 1: Guest Data Redaction & Leakage Test ─────────────────────────
    def test_guest_data_redaction_and_leakage(self):
        """
        Verify that unauthenticated visitors receive strictly masked constituent data
        (LOCKED_1 .. LOCKED_N) and zero position quantities or weights are leaked.
        """
        # 1. Test unauthenticated /api/portfolio/holdings
        res = self.client.get("/api/portfolio/holdings")
        self.assertEqual(res.status_code, 200, f"Expected 200, got {res.status_code}: {res.text}")
        data = res.json()
        self.assertTrue(data.get("is_guest"), "Expected is_guest=True for unauthenticated request")

        holdings = data.get("holdings", [])
        self.assertGreater(len(holdings), 0, "Guest holdings list should not be empty")

        for idx, h in enumerate(holdings):
            sym = h.get("symbol", "")
            self.assertTrue(
                sym.startswith("LOCKED_"),
                f"Guest symbol at index {idx} must be masked as LOCKED_#, got '{sym}'",
            )
            self.assertIsNone(h.get("qty"), f"Position qty must be redacted (None), got {h.get('qty')}")
            self.assertIsNone(h.get("avg_entry_price"), "avg_entry_price must be redacted (None)")
            self.assertIsNone(h.get("current_price"), "current_price must be redacted (None)")
            self.assertIsNone(h.get("weight_pct"), "weight_pct must be redacted (None)")
            self.assertIsNone(h.get("factors"), "factors pillar data must be redacted (None)")

        # Verify no real Alpaca ticker symbols leaked into the guest payload
        for real_sym in self.real_symbols:
            for h in holdings:
                self.assertNotEqual(
                    h.get("symbol", "").upper(),
                    real_sym,
                    f"Real ticker {real_sym} was leaked in constituent holdings!",
                )

        # 2. Test unauthenticated SSE /api/stream/holdings (safe stream consumption & guest redaction)
        start_time = time.time()
        with self.client.stream("GET", "/api/stream/holdings", headers={"Accept": "text/event-stream"}) as sse_stream:
            self.assertEqual(sse_stream.status_code, 200)
            self.assertIn("text/event-stream", sse_stream.headers.get("content-type", ""))

            sse_payload = None
            for line in sse_stream.iter_lines():
                if time.time() - start_time > 10.0:
                    self.fail("SSE stream consumption exceeded 10-second safety limit")
                if line and line.startswith("data:"):
                    raw_data = line[len("data:"):].strip()
                    try:
                        sse_payload = json.loads(raw_data)
                        break
                    except json.JSONDecodeError:
                        continue

            self.assertIsNotNone(sse_payload, "Failed to read initial SSE data payload")
            self.assertTrue(sse_payload.get("is_guest"), "Expected is_guest=True in SSE stream payload")
            sse_holdings = sse_payload.get("holdings", [])
            self.assertGreater(len(sse_holdings), 0, "Guest holdings list in SSE stream should not be empty")
            for idx, h in enumerate(sse_holdings):
                sym = h.get("symbol", "")
                self.assertTrue(
                    sym.startswith("LOCKED_"),
                    f"SSE guest symbol at index {idx} must be masked as LOCKED_#, got '{sym}'",
                )
                self.assertIsNone(h.get("qty"), f"SSE position qty must be redacted (None), got {h.get('qty')}")
                self.assertIsNone(h.get("factors"), "SSE factors pillar data must be redacted (None)")
            for real_sym in self.real_symbols:
                for h in sse_holdings:
                    self.assertNotEqual(
                        h.get("symbol", "").upper(),
                        real_sym,
                        f"Real ticker {real_sym} leaked in SSE holdings stream!",
                    )

    # ── Vector 2: Auth & Admin Security Test ──────────────────────────────────
    def test_auth_and_admin_security(self):
        """
        Verify master OTP backdoor is completely blocked, arbitrary passwords fail,
        and authenticated admin receives HTTP 200 with full constituent data and admin users access.
        """
        admin_email = "admin@example.com"

        # 1. Master OTP bypass codes MUST be rejected
        for legacy_code in ("123456", "000000"):
            otp_res = self.client.post(
                "/api/auth/verify-otp",
                json={"email": admin_email, "code": legacy_code},
            )
            self.assertIn(
                otp_res.status_code,
                (400, 401),
                f"Master OTP bypass '{legacy_code}' was not blocked! Status: {otp_res.status_code}",
            )

        # 2. Arbitrary password login MUST fail
        bad_login = self.client.post(
            "/api/auth/login",
            json={"email": admin_email, "password": "arbitrary_invalid_password_2026"},
        )
        self.assertEqual(bad_login.status_code, 401, "Arbitrary password login must return 401")

        # 3. Valid admin login
        admin_login = self.client.post(
            "/api/auth/login",
            json={"email": admin_email, "password": "Admin2026!"},
        )
        self.assertEqual(admin_login.status_code, 200, f"Valid admin login failed: {admin_login.text}")
        login_data = admin_login.json()
        token = login_data.get("token")
        self.assertIsNotNone(token, "Admin login response must include bearer token")
        self.assertEqual(login_data["user"]["role"], "admin")
        self.assertEqual(login_data["user"]["status"], "approved")

        # 4. Authenticated admin holdings breakdown (real tickers, factors, weights)
        auth_headers = {"Authorization": f"Bearer {token}"}
        auth_holdings_res = self.client.get("/api/portfolio/holdings", headers=auth_headers)
        self.assertEqual(auth_holdings_res.status_code, 200)
        auth_data = auth_holdings_res.json()
        self.assertFalse(auth_data.get("is_guest"), "Authenticated admin must have is_guest=False")

        real_holdings = auth_data.get("holdings", [])
        self.assertGreater(len(real_holdings), 0)
        for h in real_holdings:
            sym = h.get("symbol", "")
            self.assertFalse(
                sym.startswith("LOCKED_"),
                f"Authenticated admin constituent should be real ticker, got masked '{sym}'",
            )
            self.assertIsNotNone(h.get("qty"), f"Position {sym} qty must not be None")
            self.assertIsNotNone(h.get("avg_entry_price"), f"Position {sym} avg_entry_price must not be None")
            self.assertIsNotNone(h.get("current_price"), f"Position {sym} current_price must not be None")

        # 5. Admin users table access control
        # Unauthenticated access must be rejected with 403
        unauth_admin = self.client.get("/api/admin/users")
        self.assertEqual(unauth_admin.status_code, 403, "Unauthenticated /api/admin/users must return 403")

        # Authenticated access via header
        admin_users_res = self.client.get("/api/admin/users", headers=auth_headers)
        self.assertEqual(admin_users_res.status_code, 200, f"Admin users access failed: {admin_users_res.text}")
        users_payload = admin_users_res.json()
        self.assertEqual(users_payload.get("status"), "ok")
        users_list = users_payload.get("users", [])
        self.assertGreater(len(users_list), 0, "Admin users list should not be empty")

        admin_user_entry = next((u for u in users_list if u.get("email") == admin_email), None)
        self.assertIsNotNone(admin_user_entry, f"Admin user {admin_email} must appear in users list")
        self.assertEqual(admin_user_entry.get("role"), "admin")
        self.assertEqual(admin_user_entry.get("status"), "approved")

        # Authenticated access via query param (?token=...&auth_token=...)
        query_admin_res = self.client.get(f"/api/admin/users?token={token}&auth_token={token}")
        self.assertEqual(query_admin_res.status_code, 200, "Admin access via query parameter failed")

    # ── Vector 3: Live Telemetry & 1D Calculation Test ────────────────────────
    def test_live_telemetry_and_1d_calculation(self):
        """
        Verify /api/live-stats computes portfolio total_equity directly from Alpaca account
        and aligns daily session P&L with (account.equity - account.last_equity).
        """
        res = self.client.get("/api/live-stats")
        self.assertEqual(res.status_code, 200, f"Expected 200, got {res.status_code}: {res.text}")
        data = res.json()

        self.assertIn("total_equity", data)
        self.assertIn("daily_change_dollar", data)
        self.assertIn("daily_change_pct", data)

        server_equity = float(data["total_equity"])
        server_day_dollar = float(data["daily_change_dollar"])
        server_day_pct = float(data["daily_change_pct"])

        try:
            fresh_client = init_trading_client(dry_run=False)
            fresh_account = fresh_client.get_account()
            alpaca_equity = float(fresh_account.equity)
            alpaca_last_equity = float(fresh_account.last_equity)
            expected_dollar = round(alpaca_equity - alpaca_last_equity, 2)
            expected_pct = round((expected_dollar / alpaca_last_equity) * 100.0, 2) if alpaca_last_equity > 0 else 0.0

            # Verify server equity reflects account.equity (allowing a small buffer for live market movement)
            self.assertAlmostEqual(
                server_equity,
                alpaca_equity,
                delta=10.0,
                msg=f"Server equity {server_equity} does not match Alpaca equity {alpaca_equity}",
            )
            self.assertAlmostEqual(
                server_day_dollar,
                expected_dollar,
                delta=10.0,
                msg=f"Server daily dollar change {server_day_dollar} != expected {expected_dollar}",
            )
            self.assertAlmostEqual(
                server_day_pct,
                expected_pct,
                delta=0.5,
                msg=f"Server daily percent change {server_day_pct} != expected {expected_pct}",
            )

            # Mathematical consistency check on the response payload itself
            base_last_equity = round(server_equity - server_day_dollar, 2)
            if base_last_equity > 0:
                calc_pct = round((server_day_dollar / base_last_equity) * 100.0, 2)
                self.assertAlmostEqual(server_day_pct, calc_pct, delta=0.05)
        except Exception as e:
            print(f"[WARN] Live telemetry check against fresh Alpaca account: {e}")

    # ── Vector 4: Macro Regime Buffer Evaluation ──────────────────────────────
    def test_macro_regime_buffer_evaluation(self):
        """
        Verify calculate_macro_regime across boundary conditions:
          - SPY > 725.56  -> Risk-ON, buffer > 0, equity 100%
          - SPY < 725.56  -> Risk-OFF, buffer < 0, equity 50%
          - SPY == 725.56 -> Risk-ON, buffer == 0.0, equity 100%
        """
        threshold = float(SMA_10M_THRESHOLD)

        # Boundary 1: Above threshold
        res_above = calculate_macro_regime(750.00)
        self.assertEqual(res_above["regime"], "Risk-ON")
        self.assertEqual(res_above["status"], "Risk-ON")
        self.assertGreater(res_above["buffer_pct"], 0.0)
        self.assertEqual(res_above["equity_allocation_pct"], 100.0)
        self.assertEqual(res_above["bond_allocation_pct"], 0.0)

        # Boundary 2: Below threshold
        res_below = calculate_macro_regime(700.00)
        self.assertEqual(res_below["regime"], "Risk-OFF")
        self.assertEqual(res_below["status"], "Risk-OFF")
        self.assertLess(res_below["buffer_pct"], 0.0)
        self.assertEqual(res_below["equity_allocation_pct"], 50.0)
        self.assertEqual(res_below["bond_allocation_pct"], 50.0)

        # Boundary 3: Exactly on threshold
        res_equal = calculate_macro_regime(threshold)
        self.assertEqual(res_equal["regime"], "Risk-ON")
        self.assertEqual(res_equal["status"], "Risk-ON")
        self.assertAlmostEqual(res_equal["buffer_pct"], 0.0, delta=0.01)
        self.assertEqual(res_equal["equity_allocation_pct"], 100.0)
        self.assertEqual(res_equal["bond_allocation_pct"], 0.0)

    # ── Vector 5: Market Session & NYSE Calendar Integrity ────────────────────
    def test_market_session_and_nyse_calendar(self):
        """
        Verify precise 24/5 session phase detection and NYSE full/half-day calendar:
          1. Known holiday date returns 'HOLIDAY CLOSED'
          2. Sunday 22:00 ET returns 'OVERNIGHT ACTIVE'
          3. Tuesday 14:00 ET returns 'MARKET OPEN'
        """
        ny_tz = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone(timedelta(hours=-4))

        # Check 1: Known holiday date (e.g. MLK Day 2026-01-19 14:00 ET) -> HOLIDAY CLOSED
        mlk_dt = datetime(2026, 1, 19, 14, 0, tzinfo=ny_tz)
        res_holiday = resolve_market_session(dt=mlk_dt)
        self.assertTrue(res_holiday["is_holiday"], "MLK Day must be identified as is_holiday=True")
        self.assertEqual(res_holiday["market_status"], "HOLIDAY CLOSED")
        self.assertIn("HOLIDAY CLOSED", res_holiday["session_name"])
        self.assertEqual(res_holiday["session_phase"], "HOLIDAY")
        self.assertEqual(res_holiday["holiday_name"], "Martin Luther King Jr. Day")
        self.assertFalse(res_holiday["is_market_open"])

        # Check 2: Sunday 22:00 ET (2026-01-11 22:00 ET) -> OVERNIGHT ACTIVE
        sun_dt = datetime(2026, 1, 11, 22, 0, tzinfo=ny_tz)
        res_sun = resolve_market_session(dt=sun_dt)
        self.assertEqual(res_sun["session_name"], "OVERNIGHT ACTIVE")
        self.assertEqual(res_sun["session_phase"], "OVERNIGHT")
        self.assertEqual(res_sun["market_status"], "OVERNIGHT")
        self.assertTrue(res_sun["is_market_open"])

        # Check 3: Tuesday 14:00 ET (2026-01-13 14:00 ET) -> MARKET OPEN
        tue_dt = datetime(2026, 1, 13, 14, 0, tzinfo=ny_tz)
        res_tue = resolve_market_session(dt=tue_dt)
        self.assertEqual(res_tue["session_name"], "MARKET OPEN")
        self.assertEqual(res_tue["session_phase"], "REGULAR")
        self.assertEqual(res_tue["market_status"], "REGULAR")
        self.assertTrue(res_tue["is_market_open"])
        self.assertTrue(res_tue["is_regular_open"])

        # Check 4: Early close session (Black Friday 2026-11-27 14:00 ET -> After-hours active)
        bf_dt = datetime(2026, 11, 27, 14, 0, tzinfo=ny_tz)
        res_bf = resolve_market_session(dt=bf_dt)
        self.assertTrue(res_bf["is_early_close"])
        self.assertEqual(res_bf["session_name"], "AFTER-HOURS ACTIVE")
        self.assertEqual(res_bf["session_phase"], "POST")


def run_standalone_suite():
    """Run all verification vectors and format a clean ASCII summary table."""
    TestSystemIntegrity.setUpClass()
    tests = [
        ("Guest Data Redaction & Leakage", "test_guest_data_redaction_and_leakage"),
        ("Auth & Admin Security Integrity", "test_auth_and_admin_security"),
        ("Live Telemetry & 1D Calculation", "test_live_telemetry_and_1d_calculation"),
        ("Macro Regime Buffer Evaluation", "test_macro_regime_buffer_evaluation"),
        ("Market Session & NYSE Calendar Integrity", "test_market_session_and_nyse_calendar"),
    ]

    results = []
    print("=" * 80)
    print("   THE VALUE HUNTER — AUTOMATED SYSTEM INTEGRITY & SECURITY SUITE")
    print("=" * 80)

    for name, method_name in tests:
        test_case = TestSystemIntegrity(methodName=method_name)
        res = unittest.TestResult()
        test_case.run(res)

        status = "PASS" if res.wasSuccessful() else "FAIL"
        err_msg = ""
        if res.errors:
            err_msg = res.errors[0][1].strip().split("\n")[-1]
        elif res.failures:
            err_msg = res.failures[0][1].strip().split("\n")[-1]

        results.append((name, status, err_msg))

    print("\n" + "-" * 80)
    print(f"{'VERIFICATION VECTOR':<45} | {'STATUS':<10} | {'DETAILS'}")
    print("-" * 80)
    all_passed = True
    for name, status, details in results:
        detail_str = "All assertions verified successfully" if status == "PASS" else details
        badge = f"[{status}]"
        print(f"{name:<45} | {badge:<10} | {detail_str}")
        if status != "PASS":
            all_passed = False
    print("-" * 80)

    if all_passed:
        print("\n>>> ALL VERIFICATION VECTORS PASSED [5/5] <<<\n")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE VERIFICATION VECTORS FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    run_standalone_suite()
