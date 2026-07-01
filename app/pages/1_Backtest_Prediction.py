"""
Page 1 — Backtest Prediction
Reproduces model behavior on historical data by pulling a feature row directly
from the engineered features CSV (exactly what the model saw during evaluation).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_engineered_features, load_feature_names
from utils.inference import FEATURE_COLS, predict_ensemble

st.set_page_config(page_title="Backtest Prediction", page_icon="📊", layout="wide")
st.title("Backtest Prediction")
st.info(
    "This reproduces the model's behavior on historical data used in evaluation. "
    "Feature rows are pulled directly from the engineered CSV — no live computation."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    index_label = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], key="bt_index")
    index = index_label.lower()
    horizon = st.selectbox("Horizon (days)", [1, 5], key="bt_horizon")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    feat_names = load_feature_names(index)
    if feat_names != FEATURE_COLS:
        st.warning(
            f"Feature name mismatch detected!  \n"
            f"models/{index}_feature_names.txt has {len(feat_names)} features, "
            f"inference.py FEATURE_COLS has {len(FEATURE_COLS)}.  \n"
            f"Diff: {set(feat_names) ^ set(FEATURE_COLS)}"
        )
except Exception as e:
    st.error(f"Could not load feature names: {e}")
    st.stop()

try:
    df = load_engineered_features(index)
except Exception as e:
    st.error(f"Could not load engineered features CSV: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Date selector
# The last `horizon` rows cannot be used (no actual price at date+h available)
# ---------------------------------------------------------------------------

valid_df = df.iloc[: len(df) - horizon].copy()
dates_available = valid_df["Date"].dt.date.tolist()

if not dates_available:
    st.error("No valid dates available after accounting for the horizon window.")
    st.stop()

selected_date = st.date_input(
    "Select historical date",
    value=dates_available[-1],
    min_value=dates_available[0],
    max_value=dates_available[-1],
    key="bt_date",
)

# Find closest available date (date_input can return a date not in the CSV)
available_set = set(dates_available)
if selected_date not in available_set:
    # Walk backward to find the nearest trading day in the CSV
    import datetime
    candidate = selected_date
    for _ in range(10):
        candidate -= datetime.timedelta(days=1)
        if candidate in available_set:
            selected_date = candidate
            st.caption(f"Selected date is not a trading day — using {selected_date} instead.")
            break
    else:
        st.error("Could not find a valid trading day near the selected date.")
        st.stop()

row_idx = valid_df[valid_df["Date"].dt.date == selected_date].index
if len(row_idx) == 0:
    st.error(f"No data row found for {selected_date}.")
    st.stop()

row_idx = row_idx[0]
feature_row_series = df.loc[row_idx]

# ---------------------------------------------------------------------------
# Build feature dict and validate
# ---------------------------------------------------------------------------

feature_row = {}
missing_cols = []
for col in FEATURE_COLS:
    if col not in df.columns:
        missing_cols.append(col)
    else:
        val = feature_row_series[col]
        feature_row[col] = float(val) if pd.notna(val) else None

if missing_cols:
    st.error(f"Engineered features CSV is missing columns: {missing_cols}")
    st.stop()

none_features = [k for k, v in feature_row.items() if v is None]
if none_features:
    st.error(
        f"The selected row has NaN values for: {none_features}  \n"
        "Select a different date (early dates often have NaN lags)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Actual price at date + horizon
# ---------------------------------------------------------------------------

actual_row_idx = row_idx + horizon
if actual_row_idx >= len(df):
    st.error("Horizon extends beyond available data.")
    st.stop()

actual_price = float(df.loc[actual_row_idx, "Close"])
actual_date = df.loc[actual_row_idx, "Date"].date()

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"{index_label} · T+{horizon} prediction from {selected_date}")

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

error_inr  = pred_price - actual_price
error_pct  = (error_inr / actual_price) * 100 if actual_price != 0 else float("nan")
direction_correct = (pred_price > base_close) == (actual_price > base_close)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Base Close", f"₹{base_close:,.2f}", help=f"Close on {selected_date}")
col2.metric("Predicted Price", f"₹{pred_price:,.2f}", f"{pred_ret:+.3f}%")
col3.metric(
    f"Actual Price (T+{horizon})",
    f"₹{actual_price:,.2f}",
    help=f"Close on {actual_date}",
)
col4.metric(
    "Error",
    f"₹{error_inr:+,.2f}",
    f"{error_pct:+.2f}%",
    delta_color="inverse",
)

st.markdown(
    f"**Direction:** {'Correct' if direction_correct else 'Incorrect'} &nbsp;|&nbsp; "
    f"Predicted {'up' if pred_price >= base_close else 'down'}, "
    f"Actual {'up' if actual_price >= base_close else 'down'}."
)

st.divider()

# Base model predictions breakdown
with st.expander("Base model predictions (% return)"):
    bm = result["base_model_preds"]
    bm_df = pd.DataFrame(
        [{"Model": k, "Predicted Return (%)": f"{v:+.4f}"} for k, v in bm.items()]
    )
    st.dataframe(bm_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Meta (Ridge) output: {pred_ret:+.4f}% → ₹{pred_price:,.2f}"
    )

# Feature row used
with st.expander("Feature row used for inference"):
    frow_df = pd.DataFrame(
        [{"Feature": k, "Value": f"{v:.4f}"} for k, v in feature_row.items()]
    )
    st.dataframe(frow_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Source: data/processed/{index}_engineered_features.csv, row {row_idx} ({selected_date})"
    )
