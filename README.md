# AI Portfolio Optimizer

## Workflow Overview

This project delivers:
1. Preprocessing and exploring market data
2. Developing and comparing forecasting models (ARIMA vs. LSTM)
3. Producing a 12‑month TSLA forecast
4. Constructing an optimized portfolio based on the forecast
5. Backtesting against a benchmark

---

## Project Structure

```
financial_portfolio_optimization/
├─ data/
│  ├─ raw/financial_data.csv
│  ├─ processed/adj_close.csv
│  ├─ processed/tsla_12month_forecast.csv
├─ models/lstm_tsla_forecast_model.keras
├─ notebooks/
├─ scripts/
│  ├─ data_ingestion.py
│  ├─ model_training.py
│  └─ portfolio_analysis.py
├─ reports/figures/
├─ README.md
└─ requirements.txt
```

---

## Environment

Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

**Task 1:**  
```bash
python scripts/data_ingestion.py
```
**Tasks 2 & 3:**  
```bash
python scripts/model_training.py
```
**Tasks 4 & 5:**  
```bash
python scripts/portfolio_analysis.py
```
Or use Jupyter notebooks in `/notebooks`.

---

## Task Summaries

### Task 1 — Data Preprocessing and Exploration

- Assets: TSLA, BND, SPY (2015‑07‑01 to 2025‑07‑31, daily adjusted close)
- No missing values in processed data
- TSLA shows high post‑2020 growth and volatility; BND stable; SPY steady uptrend
- Stationarity: Prices non‑stationary, daily returns stationary
- Risk metrics: TSLA shows highest volatility and risk

---

### Task 2 — Forecasting Models (TSLA)

- ARIMA (classical) vs. LSTM (deep learning) for daily TSLA prices
- LSTM outperformed ARIMA (lower MAE, RMSE, MAPE)
- LSTM selected for final forecast

---

### Task 3 — 12‑Month TSLA Forecast

- LSTM predicts a downward trend ($314 → $144)
- Early volatility, smoothing over longer horizon
- Caveats: High uncertainty for long forecasts

---

### Task 4 — Portfolio Optimization

- Efficient frontier via MPT using TSLA forecast, historical BND/SPY
- Maximum Sharpe Ratio Portfolio: ~96% SPY, ~4% BND, 0% TSLA
- TSLA excluded due to bearish forecast

---

### Task 5 — Backtesting

- Backtest period: 2024‑08‑01 to 2025‑07‑31
- Strategy outperformed 60/40 benchmark (17.69% vs 12.47% returns)
- Slightly lower Sharpe, but higher total return

---

## Reproducibility & Assumptions

- Time-based splits, 252 trading days/year, rf = 0%
- No rebalancing, no transaction costs, perfect execution assumed
- Minor run-to-run variance possible due to random seeds