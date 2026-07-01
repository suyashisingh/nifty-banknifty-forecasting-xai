"""
Page 2 — Live Prediction (Experimental)
Fetches recent OHLCV via yfinance, computes indicators matching the training
pipeline, forward-fills macro/sentiment from last known historical values,
then runs ensemble inference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_engineered_features,
    load_fii_dii,
    load_feature_names,
    load_rbi_repo,
)
from utils.indicators import compute_features
from utils.inference import FEATURE_COLS, predict_ensemble

st.set_page_config(page_title="Live Prediction", page_icon="⚡", layout="wide")
st.title("Live Prediction")

st.warning(
    "**Experimental — illustrative only, not validated against paper metrics.**  \n"
    "Macro and sentiment features (FII/DII flows, Repo Rate, Reddit Sentiment) are "
    "forward-filled from the last known historical value in the project CSV files, "
    "not fetched live. The 'as of' date for each is displayed below."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    index_label = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], key="lp_index")
    index = index_label.lower()
    horizon = st.selectbox("Horizon (days)", [1, 5], key="lp_horizon")

TICKER_MAP = {"nifty": "^NSEI", "banknifty": "^NSEBANK"}
ticker = TICKER_MAP[index]

# ---------------------------------------------------------------------------
# Step 1: Fetch OHLCV
# ---------------------------------------------------------------------------

st.subheader("Step 1 — Fetch OHLCV")

@st.cache_data(ttl=3600)
def fetch_ohlcv(ticker: str, days: int = 90) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), interval="1d", progress=False)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if c[1] == "" else c[0] for c in raw.columns]
    raw = raw.rename(columns={"Date": "Date"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in raw.columns:
            raw[col] = np.nan
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

with st.spinner(f"Fetching last 90 days of {ticker} from yfinance..."):
    try:
        ohlcv = fetch_ohlcv(ticker)
    except Exception:
        st.warning("Live market data is temporarily unavailable. Please try again in a moment.")
        st.stop()

if ohlcv.empty:
    st.warning(
        f"Live market data is temporarily unavailable for {ticker}. "
        "Please try again in a moment."
    )
    st.stop()

st.success(
    f"Fetched {len(ohlcv)} trading days: "
    f"{ohlcv['Date'].iloc[0].date()} to {ohlcv['Date'].iloc[-1].date()}"
)
_ohlcv_latest = ohlcv["Date"].iloc[-1].date()
if _ohlcv_latest < datetime.date.today():
    st.caption(f"Latest available trading day: {_ohlcv_latest}")

if len(ohlcv) < 20:
    st.error(
        "Fewer than 20 rows returned — not enough data to compute reliable indicators. "
        "Indicators require at least 20 rows for SMA/Bollinger Bands."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 2: Compute technical indicators + lags
# ---------------------------------------------------------------------------

st.subheader("Step 2 — Compute Technical Indicators")

with st.spinner("Computing indicators (matching training pipeline)..."):
    try:
        df_ind = compute_features(ohlcv.copy())
    except Exception as e:
        st.error(f"Indicator computation failed: {e}")
        st.stop()

INDICATOR_COLS = [
    "SMA", "EMA", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
]
missing_ind = [c for c in INDICATOR_COLS if c not in df_ind.columns]
if missing_ind:
    st.error(
        f"Indicator computation did not produce expected columns: {missing_ind}  \n"
        "This may indicate a version mismatch with the `ta` library."
    )
    st.stop()

st.success("Indicators computed successfully.")

last_row = df_ind.iloc[-1]
latest_date = last_row["Date"].date() if hasattr(last_row["Date"], "date") else last_row["Date"]

# ---------------------------------------------------------------------------
# Step 3: Forward-fill macro and sentiment from historical CSVs
# ---------------------------------------------------------------------------

st.subheader("Step 3 — Forward-fill Macro & Sentiment Features")

def get_last_known(index: str) -> dict:
    """Get the last non-null values for FII_Net, DII_Net, Repo_Rate, Reddit_Sentiment
    from the engineered features CSV (already processed and merged)."""
    eng = load_engineered_features(index)
    eng_sorted = eng.sort_values("Date")
    macro_cols = ["FII_Net", "DII_Net", "Repo_Rate", "Reddit_Sentiment"]
    result = {}
    dates  = {}
    for col in macro_cols:
        if col in eng_sorted.columns:
            valid = eng_sorted[eng_sorted[col].notna() & (eng_sorted[col] != 0.0)]
            if not valid.empty:
                result[col] = float(valid[col].iloc[-1])
                dates[col]  = valid["Date"].iloc[-1].date()
            else:
                result[col] = 0.0
                dates[col]  = None
        else:
            result[col] = 0.0
            dates[col]  = None
    return result, dates

with st.spinner("Reading last known macro/sentiment values..."):
    try:
        macro_vals, macro_dates = get_last_known(index)
    except Exception as e:
        st.error(f"Could not load macro/sentiment data: {e}")
        st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("FII_Net (Cr)", f"{macro_vals['FII_Net']:,.2f}",
            help=f"As of {macro_dates.get('FII_Net', 'unknown')}")
col2.metric("DII_Net (Cr)", f"{macro_vals['DII_Net']:,.2f}",
            help=f"As of {macro_dates.get('DII_Net', 'unknown')}")
col3.metric("Repo Rate (%)", f"{macro_vals['Repo_Rate']:.2f}",
            help=f"As of {macro_dates.get('Repo_Rate', 'unknown')}")
col4.metric("Reddit Sentiment", f"{macro_vals['Reddit_Sentiment']:.4f}",
            help=f"As of {macro_dates.get('Reddit_Sentiment', 'unknown')}")

with st.expander("Staleness detail"):
    staleness_rows = [
        {"Feature": k, "Last Known Value": v, "As of Date": str(macro_dates.get(k, "N/A"))}
        for k, v in macro_vals.items()
    ]
    st.dataframe(pd.DataFrame(staleness_rows), use_container_width=True, hide_index=True)
    st.caption(
        "These values are forward-filled — they may be days, weeks, or months stale. "
        "FII/DII data in this project is monthly-granularity."
    )

# ---------------------------------------------------------------------------
# Step 4: Build feature dict and run inference
# ---------------------------------------------------------------------------

st.subheader("Step 4 — Run Inference")

# Validate that all OHLCV + indicator columns are present and non-NaN in last row
ohlcv_ind_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA", "EMA", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
]

feature_row = {}
bad_cols = []
for col in ohlcv_ind_cols:
    val = last_row.get(col) if hasattr(last_row, "get") else last_row[col]
    if pd.isna(val):
        bad_cols.append(col)
    else:
        feature_row[col] = float(val)

if bad_cols:
    st.error(
        f"The following features are NaN in the latest row ({latest_date}): {bad_cols}  \n"
        "This typically happens when the trailing window is too short for lag features. "
        "Try refreshing — yfinance sometimes returns fewer rows than expected."
    )
    st.stop()

# Add macro/sentiment
for col, val in macro_vals.items():
    feature_row[col] = val

# Final NaN guard
none_feats = [f for f in FEATURE_COLS if feature_row.get(f) is None or np.isnan(feature_row.get(f, float("nan")))]
if none_feats:
    st.error(f"Cannot run inference — features still missing/NaN after all steps: {none_feats}")
    st.stop()

try:
    result = predict_ensemble(index, horizon, feature_row)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Inference failed: {e}")
    st.stop()

pred_price = result["predicted_price"]
pred_ret   = result["predicted_return_pct"]
base_close = result["base_close"]

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"Result — {index_label} T+{horizon} from {latest_date}")

col1, col2, col3 = st.columns(3)
col1.metric("Current Close", f"₹{base_close:,.2f}", help=f"Close on {latest_date}")
col2.metric(
    f"Predicted Close (T+{horizon})",
    f"₹{pred_price:,.2f}",
    f"{pred_ret:+.3f}%",
)
col3.metric(
    "Direction",
    "Up" if pred_price >= base_close else "Down",
    delta=f"{pred_ret:+.3f}%",
    delta_color="normal" if pred_price >= base_close else "inverse",
)

with st.expander("Base model predictions (% return)"):
    bm = result["base_model_preds"]
    bm_df = pd.DataFrame(
        [{"Model": k, "Predicted Return (%)": f"{v:+.4f}"} for k, v in bm.items()]
    )
    st.dataframe(bm_df, use_container_width=True, hide_index=True)

with st.expander("Full feature row used"):
    frow_df = pd.DataFrame(
        [{"Feature": k, "Value": f"{v:.4f}"} for k, v in feature_row.items()]
    )
    st.dataframe(frow_df, use_container_width=True, hide_index=True)

with st.expander("Recent OHLCV (last 10 rows)"):
    st.dataframe(
        ohlcv.tail(10).set_index("Date").style.format("{:.2f}"),
        use_container_width=True,
    )
