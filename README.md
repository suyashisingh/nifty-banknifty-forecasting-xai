# NIFTY & BANKNIFTY Forecasting with XAI

A full-stack ML web application for multi-horizon price forecasting of India's two major stock indices — NIFTY 50 and BANKNIFTY — with explainable AI (XAI) interpretability built in.

Live demo: [nifty-banknifty-forecasting-xai.streamlit.app](https://nifty-banknifty-forecasting-xai.streamlit.app)

---

## What it does

- **Short-horizon forecasting (T+1, T+5):** A stacked ensemble of XGBoost, LightGBM, CatBoost, and AdaBoost with a Ridge meta-learner predicts next-day and next-week closing prices for NIFTY and BANKNIFTY.
- **Long-horizon forecasting (T+30):** Facebook Prophet augmented with exogenous regressors (FII/DII flows, RBI repo rate, market sentiment) produces a 30-day ahead forecast with confidence intervals.
- **Explainability (XAI):** SHAP, LIME, and ELI5 provide feature-level interpretability for all ensemble configurations — showing which features drive each prediction and by how much.
- **Live market data:** Fetches real-time OHLCV data via yfinance and computes technical indicators on the fly for live predictions.
- **Sentiment analysis:** Live stock news sentiment scoring via VADER on any NSE-listed ticker symbol.

---

## Tech stack

| Layer | Tools |
|---|---|
| ML models | XGBoost, LightGBM, CatBoost, AdaBoost, Ridge, Facebook Prophet |
| XAI | SHAP, LIME, ELI5 |
| Features | TA-Lib (SMA, EMA, RSI, MACD, Bollinger Bands, ATR), FII/DII flows, RBI repo rate, Reddit sentiment (VADER + FinBERT) |
| Frontend | Streamlit |
| Data | yfinance, NSE India, RBI, Reddit (r/IndiaInvestments) |
| Language | Python 3.12 |

---

## App pages

| Page | Description |
|---|---|
| **Home** | Live NIFTY/BANKNIFTY ticker, stock news sentiment analyzer, quick prediction generator (T+1 / T+5 / T+30) |
| **Backtest Prediction** | Select any historical date and see what the model would have predicted vs. what actually happened — full feature row transparency |
| **Live Prediction** | End-to-end live inference: fetches today's market data, computes indicators, runs the ensemble, shows next-day prediction |
| **Prophet Forecast** | 30-day ahead forecast chart with confidence band (yhat_lower / yhat_upper) |
| **XAI Explainability** | SHAP beeswarm and bar plots, LIME local explanations, ELI5 permutation importance — for all 4 model configurations |

---

## Model performance

| Index | Model | Horizon | RMSE | MAPE | DA |
|---|---|---|---|---|---|
| NIFTY | Stacked Ensemble | T+1 | 173.81 | 0.64% | 52.85% |
| NIFTY | Stacked Ensemble | T+5 | 385.89 | 1.56% | 58.10% |
| NIFTY | Prophet | T+30 | 742.18 | 3.69% | 62.07% |
| BANKNIFTY | Stacked Ensemble | T+1 | 470.23 | 0.78% | 52.91% |
| BANKNIFTY | Stacked Ensemble | T+5 | 1051.64 | 1.93% | 57.37% |
| BANKNIFTY | Prophet | T+30 | 2118.74 | 4.54% | 51.72% |

---

## Feature engineering (23 features)

- **OHLCV (5):** Open, High, Low, Close, Volume
- **Technical indicators (9):** SMA, EMA, RSI, MACD, MACD Signal, Bollinger Bands (Upper/Middle/Lower), ATR
- **Lag features (5):** Close at T−1, T−2, T−3, T−5, T−10
- **Macroeconomic (3):** FII Net flows, DII Net flows, RBI Repo Rate
- **Sentiment (1):** Reddit sentiment score (VADER + FinBERT combined)

---

## Run locally

```bash
git clone https://github.com/suyashisingh/nifty-banknifty-forecasting-xai.git
cd nifty-banknifty-forecasting-xai
pip install -r requirements-deploy.txt
streamlit run app/Home.py
```

---

## Project structure

```
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Backtest_Prediction.py
│   │   ├── 2_Live_Prediction.py
│   │   ├── 3_Prophet_Forecast.py
│   │   └── 4_XAI_Explainability.py
│   └── utils/
│       ├── data_loader.py
│       ├── indicators.py
│       └── inference.py
├── models/          # Trained model artifacts (.pkl)
├── data/            # Processed feature CSVs
├── src/             # Training pipeline scripts
└── requirements-deploy.txt
```

---

## Author

**Suyashi Singh**
School of Computer Engineering, Manipal Institute of Technology, MAHE
