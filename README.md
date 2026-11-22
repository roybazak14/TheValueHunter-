# The Value Hunter

A research sandbox for quantitative multi-factor equity strategy modeling, data engineering, and simulated forward-testing.

> **Disclaimer:** This project is strictly for academic research and educational exploration. It is **not** financial advice, an investment recommendation, or an offer of financial services. No real capital is managed or deployed; all live testing occurs via broker demo/paper accounts.

---

## Overview

The Value Hunter explores a systematic, rules-based investment workflow designed to balance factor exposure (quality, stability, and momentum) with defensive macro-regime filtering.

Key areas of research include:
- **Macro Overlay Modeling:** Regime detection using broad market moving averages (e.g., SPY 10-month simple moving average) to throttle equity exposure and rotate into defensive assets (such as Treasury ETFs) during prolonged market downturns.
- **Factor Scoring:** Cross-sectional ranking combining balance-sheet quality (free cash flow generation, conservative leverage), price momentum, and low-volatility stability metrics.
- **Execution Simulation:** Automated paper-trading integration using Alpaca's paper API to observe execution mechanics, slippage, and real-time portfolio drift.
- **Research Dashboard:** A local interactive web application providing telemetry, regime diagnostics, and factor inspection.

---

## Architecture & Tech Stack

- **Core Engine:** Python 3.10+
- **Data Engineering:** `pandas`, `numpy`, `yfinance`, SEC EDGAR point-in-time public feeds
- **API Server & Telemetry:** FastAPI, Uvicorn, SQLite
- **Broker Interface:** `alpaca-py` (configured strictly for paper trading)
- **Frontend / Dashboard:** Vanilla HTML5, Tailwind CSS, Chart.js

---

## Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/roybazak14/TheValueHunter.git
cd TheValueHunter
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure paper-trading credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Alpaca paper API credentials:
```env
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_PAPER=True
```

### 3. Running the Components

- **Generate Current Portfolio Signals:**
  ```bash
  python get_current_signals.py
  ```

- **Run Rebalance Simulation (Dry-Run):**
  ```bash
  python alpaca_execution.py --dry-run
  ```

- **Launch Research Dashboard & Local API Server:**
  ```bash
  python api_server.py
  ```
  Open `index.html` in your browser to inspect the interactive dashboard.

---

## Honest Research Limitations & Caveats

Rigorous quantitative research requires acknowledging pitfalls and model limitations:

1. **Initial Backtest Lookahead Bias:**
   Initial backtest models produced unrealistically high results due to data-timing mismatches (specifically, aligning fiscal period dates with market disclosure dates, lookahead bias in historical index constituent mapping, and survivorship bias). These inflated historical returns were invalidated upon stricter data audit.
2. **Shift to Live Forward Testing:**
   Because historical point-in-time fundamental databases with exact SEC acceptance timestamps are expensive and difficult to reconstruct cleanly with free data, the model is now evaluated primarily via live forward testing in a demo paper account. Forward testing guarantees zero lookahead bias by construction.
3. **Free Data Dependencies:**
   The pipeline relies on free, publicly scraped data sources (Yahoo Finance, public SEC EDGAR). Free data is subject to unannounced API changes, rate limits, dividend adjustment discrepancies, and missing historical delistings.
4. **Market Microstructure & Slippage:**
   Paper trading simulations do not fully replicate real-world liquidity constraints, market impact on smaller-cap names, borrow fees, or partial fills during volatile market sessions.
