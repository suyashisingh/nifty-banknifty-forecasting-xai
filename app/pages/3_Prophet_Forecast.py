"""
Page 3 — Prophet Forecast
30-day ahead price forecast with yhat_lower/yhat_upper confidence band.
Toggle between the pre-computed CSV (fast, always available) and a live
re-run of the saved Prophet model using forward-filled regressors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from utils.data_loader import load_engineered_features, load_prophet_forecast
from utils.inference import predict_prophet_live

st.set_page_config(page_title="Prophet Forecast", page_icon="🔮", layout="wide")
st.title("Prophet Forecast — T+30")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    index_label = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], key="pf_index")
    index = index_label.lower()

mode = st.radio(
    "Forecast mode",
    ["Pre-computed forecast (fast)", "Re-run Prophet model (live)"],
    horizontal=True,
    key="pf_mode",
)

# ---------------------------------------------------------------------------
# Load forecast data
# ---------------------------------------------------------------------------

forecast_df = None
mode_note   = ""

if mode.startswith("Pre-computed"):
    try:
        forecast_df = load_prophet_forecast(index)
        import datetime as _dt_mod
        _last_date_s = forecast_df["ds"].max().date()
        _gap_days_s  = (_dt_mod.date.today() - _last_date_s).days
        staleness    = {"last_date": _last_date_s, "gap_days": _gap_days_s, "is_stale": _gap_days_s > 90}
        if staleness["is_stale"]:
            st.warning(
                f"This forecast is based on data as of **{staleness['last_date']}** "
                f"and is now **{staleness['gap_days']} days old**. "
                "Prophet's trend extrapolation becomes unreliable this far past "
                "training data — treat this as illustrative, not current. "
                "Will self-update once Prophet is refit on recent data and/or "
                "the 'Re-run' option is used.",
                icon="⚠️",
            )
        mode_note = (
            f"Showing pre-computed forecast from "
            f"`models/{index}_prophet_forecast.csv` — "
            f"generated at training time from data ending "
            f"~2024. Will become stale over time."
        )
    except Exception as e:
        st.error(f"Could not load pre-computed forecast: {e}")
        st.stop()

else:
    # Re-run using saved model + forward-filled regressors from engineered CSV
    st.info(
        "Loading Prophet model and forward-filling regressors from the last known "
        "row of the engineered features CSV. This may take 10–30 seconds."
    )
    try:
        eng = load_engineered_features(index)
        eng_sorted = eng.sort_values("Date")
        last_known = eng_sorted.iloc[-1]
        regressors = {
            "Repo_Rate":        float(last_known.get("Repo_Rate", 0) or 0),
            "FII_Net":          float(last_known.get("FII_Net", 0) or 0),
            "DII_Net":          float(last_known.get("DII_Net", 0) or 0),
            "Reddit_Sentiment": float(last_known.get("Reddit_Sentiment", 0) or 0),
        }
        last_date = eng_sorted["Date"].iloc[-1].date()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Repo_Rate", f"{regressors['Repo_Rate']:.2f}%")
        col2.metric("FII_Net", f"{regressors['FII_Net']:,.2f}")
        col3.metric("DII_Net", f"{regressors['DII_Net']:,.2f}")
        col4.metric("Reddit Sentiment", f"{regressors['Reddit_Sentiment']:.4f}")
        st.caption(f"All regressors forward-filled from engineered CSV last row ({last_date}).")

    except Exception as e:
        st.error(f"Could not load engineered features for regressor forward-fill: {e}")
        st.stop()

    with st.spinner("Running Prophet model.predict() for next 30 business days..."):
        try:
            forecast_df = predict_prophet_live(index, regressors)
            mode_note = (
                "Live re-run of the saved Prophet model. Regressors forward-filled "
                f"from last known value ({last_date})."
            )
        except Exception as e:
            st.error(
                f"Prophet inference failed: {e}  \n"
                "This may occur if the Prophet model pickle was saved with a different "
                "Prophet/Stan version. Try the pre-computed forecast mode instead."
            )
            st.stop()

if forecast_df is None or forecast_df.empty:
    st.error("No forecast data available.")
    st.stop()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

st.subheader(f"{index_label} — 30-Day Prophet Forecast")
if mode_note:
    st.caption(mode_note)

try:
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper bound",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(width=0),
        name="95% Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines+markers",
        line=dict(color="rgb(99, 110, 250)", width=2),
        marker=dict(size=5),
        name="Forecast (yhat)",
    ))

    fig.update_layout(
        title=f"{index_label} Prophet Forecast — Next 30 Business Days",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

except ImportError:
    # Fallback to matplotlib if plotly not available
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df["ds"], forecast_df["yhat"], label="yhat", color="steelblue", linewidth=2)
    ax.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.2,
        color="steelblue",
        label="95% CI",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.set_title(f"{index_label} Prophet Forecast — Next 30 Business Days")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Forecast table
# ---------------------------------------------------------------------------

with st.expander("Forecast values (table)"):
    display_df = forecast_df.copy()
    display_df["ds"] = display_df["ds"].dt.date
    display_df = display_df.rename(columns={
        "ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower (95%)", "yhat_upper": "Upper (95%)"
    })
    st.dataframe(
        display_df.style.format({
            "Forecast": "₹{:,.2f}",
            "Lower (95%)": "₹{:,.2f}",
            "Upper (95%)": "₹{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Forecast Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Start Date",     str(forecast_df["ds"].iloc[0].date()))
col2.metric("End Date",       str(forecast_df["ds"].iloc[-1].date()))
col3.metric("Min Forecast",   f"₹{forecast_df['yhat'].min():,.2f}")
col4.metric("Max Forecast",   f"₹{forecast_df['yhat'].max():,.2f}")

st.caption(
    "Published paper metrics (T+30): "
    "NIFTY Prophet RMSE=742.18, MAPE=3.69%, DA=62.07% · "
    "BANKNIFTY Prophet RMSE=2118.74, MAPE=4.54%, DA=51.72%"
)
