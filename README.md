# Hybrid Stacked Ensemble and Prophet-Based Multi-Horizon Forecasting of NIFTY and BANKNIFTY with XAI Interpretability

**Authors:** Suyashi Singh, Diana Olivia  
**Institution:** School of Computer Engineering, Manipal Institute of Technology, MAHE  
**Conference:** CISCOM 2026 (Springer LNCS)

---

## Overview

This repository contains the complete machine learning pipeline for hybrid multi-horizon forecasting of Indian stock market indices (NIFTY 50 and BANKNIFTY) using a stacked boosting ensemble for short-term prediction (T+1, T+5) and Facebook Prophet augmented with exogenous regressors for long-term trend estimation (T+30), with three-layer XAI interpretability via SHAP, LIME, and ELI5.

---

## Repository Structure

### `src/`

| File | Description |
|------|-------------|
| `data_collection.py` | NIFTY/BANKNIFTY OHLCV via yfinance (2010-2024) |
| `data_collection_fii_dii.py` | FII/DII institutional flow data |
| `data_collection_reddit.py` | Reddit sentiment (r/IndiaInvestments) |
| `data_preprocessing.py` | Merges OHLCV, macro, sentiment; handles missing values |
| `feature_engineering.py` | Constructs 23 features |
| `rescore_reddit_finbert.py` | VADER + FinBERT dual-model sentiment rescoring |
| `train_nifty_short.py` | Stacked ensemble for NIFTY T+1/T+5 |
| `train_banknifty_short.py` | Stacked ensemble for BANKNIFTY T+1/T+5 |
| `train_nifty_prophet.py` | Prophet with exogenous regressors for NIFTY T+30 |
| `train_banknifty_prophet.py` | Prophet with exogenous regressors for BANKNIFTY T+30 |
| `baseline_vs_naive.py` | Ensemble vs XGBoost vs Naive Persistence evaluation |
| `residual_analysis.py` | Prophet residual diagnostics |
| `final_evaluation.py` | Consolidated evaluation report |
| `xai.py` | SHAP, LIME, ELI5 for all 4 configurations |
| `combine_xai_figures.py` | Combines XAI plots into paper-ready grids |
| `plot_rmse_comparison.py` | RMSE comparison chart |
| `plot_predicted_vs_actual.py` | Predicted vs actual prices |
| `plot_fig5_prophet_forecast.py` | Prophet forecast with eval window |
| `plot_fig6_residual_combined.py` | Residual diagnostics |

---

## Features

**23 features total across 5 categories:**

- **Technical indicators (9):** SMA, EMA, RSI, MACD, MACD_Signal, BB_Upper, BB_Middle, BB_Lower, ATR
- **OHLCV (5):** Open, High, Low, Close, Volume
- **Closing price lags (5):** Close_Lag_1, Close_Lag_2, Close_Lag_3, Close_Lag_5, Close_Lag_10
- **Macroeconomic (3):** FII_Net, DII_Net, Repo_Rate
- **Sentiment (1):** Reddit_Sentiment (VADER + FinBERT combined)

---

## Key Results

| Index | Model | Horizon | RMSE | MAPE | DA |
|-------|-------|---------|------|------|----|
| NIFTY | Stacked Ensemble | Next-day (T+1) | 173.81 | 0.64% | 52.85% |
| NIFTY | Stacked Ensemble | Next-week (T+5) | 385.89 | 1.56% | 58.10% |
| NIFTY | Prophet | Next-month (T+30) | 742.18 | 3.69% | 62.07% |
| BANKNIFTY | Stacked Ensemble | Next-day (T+1) | 470.23 | 0.78% | 52.91% |
| BANKNIFTY | Stacked Ensemble | Next-week (T+5) | 1051.64 | 1.93% | 57.37% |
| BANKNIFTY | Prophet | Next-month (T+30) | 2118.74 | 4.54% | 51.72% |

---

## Running the Pipeline

Run scripts in the following order:

```bash
python src/data_collection.py
python src/data_collection_fii_dii.py
python src/data_collection_reddit.py
python src/rescore_reddit_finbert.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_nifty_short.py
python src/train_banknifty_short.py
python src/train_nifty_prophet.py
python src/train_banknifty_prophet.py
python src/baseline_vs_naive.py
python src/residual_analysis.py
python src/xai.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Data Sources

- **OHLCV:** NSE India (https://www.nseindia.com) via yfinance (https://github.com/ranaroussi/yfinance)
- **FII/DII Flows:** NSE India (https://www.nseindia.com)
- **RBI Repo Rate:** Reserve Bank of India (https://www.rbi.org.in)
- **Reddit Sentiment:** r/IndiaInvestments (https://www.reddit.com/r/IndiaInvestments)

---

## Notes

- Reddit API credentials required — replace `YOUR_REDDIT_CLIENT_ID` and `YOUR_REDDIT_CLIENT_SECRET` in `src/data_collection_reddit.py`
- Trained model artifacts (`.pkl`) and raw data files are excluded due to size. Run the pipeline sequentially to reproduce all results.

---

## Citation

Singh, S., Olivia, D.: Hybrid Stacked Ensemble and Prophet-Based Multi-Horizon Forecasting of NIFTY and BANKNIFTY with XAI Interpretability. In: CISCOM 2026, Springer LNCS.
