"""
Home page — Ticker bar · News Sentiment Analyzer · NIFTY/BANKNIFTY Predictor
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_baseline_csv, load_engineered_features, load_prophet_forecast
from utils.indicators import compute_features
from utils.inference import FEATURE_COLS, predict_ensemble, predict_prophet_live

st.set_page_config(
    page_title="NIFTY/BANKNIFTY Forecasting",
    page_icon="📈",
    layout="wide",
)

# ── MAE lookup for ±range on ensemble predictions ──────────────────────────
MAE_LOOKUP: dict = {}
try:
    _bl = load_baseline_csv()
    for _, _r in _bl[_bl["Model"] == "Ensemble"].iterrows():
        MAE_LOOKUP[(_r["Index"].lower(), int(_r["Horizon"]))] = float(_r["MAE"])
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — TICKER BAR
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def _get_index_prices() -> dict:
    import yfinance as yf
    out = {}
    for label, sym in [("NIFTY", "^NSEI"), ("BANKNIFTY", "^NSEBANK")]:
        try:
            fi    = yf.Ticker(sym).fast_info
            price = float(fi.last_price)
            prev  = float(fi.previous_close)
            chg   = price - prev
            pct   = (chg / prev * 100) if prev else 0.0
            out[label] = {"price": price, "change": chg, "pct": pct}
        except Exception:
            out[label] = None
    return out


try:
    _prices = _get_index_prices()
except Exception:
    _prices = {}

_parts = []
for _lbl in ["NIFTY", "BANKNIFTY"]:
    _d = _prices.get(_lbl)
    if _d:
        _clr   = "#00c176" if _d["change"] >= 0 else "#ff4b4b"
        _arrow = "▲" if _d["change"] >= 0 else "▼"
        _parts.append(
            f'<span style="color:#f0f0f0;font-weight:700;font-size:1.9em">{_lbl}:&nbsp;'
            f'₹{_d["price"]:,.2f}</span>'
            f'&nbsp;<span style="color:{_clr};font-weight:700;font-size:1.9em">'
            f'{_arrow}&nbsp;{abs(_d["change"]):,.2f}&nbsp;({_d["pct"]:+.2f}%)</span>'
        )
    else:
        _parts.append(
            f'<span style="color:#888;font-size:1.9em">{_lbl}: unavailable</span>'
        )

_sep = '&nbsp;&nbsp;&nbsp;<span style="color:#444;font-size:2em">|</span>&nbsp;&nbsp;&nbsp;'
st.markdown(
    '<div style="background:#1A1D24;padding:18px 32px;border-radius:8px;'
    'margin-bottom:20px;border:1px solid #2D3139;text-align:center">'
    + _sep.join(_parts)
    + "</div>",
    unsafe_allow_html=True,
)

st.title("Hybrid Stacked Ensemble & Prophet Forecasting")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — STOCK NEWS SENTIMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Stock News Sentiment Analyzer")
st.caption("Score live headlines via VADER — independent of the trained price models.")

_col_input, _col_btn = st.columns([5, 1])
with _col_input:
    news_symbol = st.text_input(
        "Enter Stock Name or Symbol",
        placeholder="e.g. RELIANCE.NS  or  ^NSEI  or  TCS.NS",
        label_visibility="collapsed",
    )
with _col_btn:
    analyze_clicked = st.button("Analyze", use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_news(symbol: str) -> list:
    import yfinance as yf
    try:
        raw = yf.Ticker(symbol).news or []
        headlines = []
        for item in raw[:5]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        return [f"__ERROR__{e}"]


def _render_sentiment(headlines: list, symbol: str) -> None:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    BADGE = {
        "Positive": ("#00c176", "✔", "Positive"),
        "Neutral":  ("#f0a500", "⚠", "Neutral"),
        "Negative": ("#ff4b4b", "✖", "Negative"),
    }

    for hl in headlines:
        if hl.startswith("__ERROR__"):
            st.warning(f"Could not fetch news: {hl[9:]}")
            return
        scores = sia.polarity_scores(hl)
        c = scores["compound"]
        if c >= 0.05:
            sentiment = "Positive"
        elif c <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        color, icon, label = BADGE[sentiment]
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;'
            f'padding:8px 12px;margin-bottom:6px;background:#1E2329;border-radius:6px;'
            f'border-left:4px solid {color}">'
            f'<span style="background:{color};color:#fff;font-weight:700;'
            f'font-size:0.78em;padding:2px 8px;border-radius:4px;white-space:nowrap;'
            f'min-width:72px;text-align:center">{icon} {label}</span>'
            f'<span style="color:#e0e0e0;flex:1">{hl}</span>'
            f'<span style="color:#aaa;font-size:0.85em;white-space:nowrap">({c:+.3f})</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


if analyze_clicked and news_symbol.strip():
    sym = news_symbol.strip()
    with st.spinner(f"Fetching news for {sym}…"):
        headlines = _fetch_news(sym)
    if not headlines:
        st.info(
            f"No news found for **{sym}**. "
            "Check the symbol (e.g. add `.NS` for NSE-listed stocks: `RELIANCE.NS`)."
        )
    else:
        st.caption(f"Top {len(headlines)} headlines for **{sym}** — scored with VADER")
        _render_sentiment(headlines, sym)
elif analyze_clicked:
    st.warning("Please enter a stock symbol first.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — NIFTY & BANKNIFTY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("NIFTY & BANKNIFTY Predictor")

_pred_col1, _pred_col2 = st.columns([3, 1])
with _pred_col1:
    pred_index_label = st.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY"],
        key="home_pred_index",
        label_visibility="visible",
    )
with _pred_col2:
    st.write("")  # vertical alignment spacer
    generate_btn = st.button("Generate Predictions", use_container_width=True, type="primary")

pred_index = pred_index_label.lower()
TICKER_MAP = {"nifty": "^NSEI", "banknifty": "^NSEBANK"}


# ── Live OHLCV + indicators (mirrors Page 2 pipeline) ──────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlcv(ticker: str, days: int = 90) -> pd.DataFrame:
    import yfinance as yf, datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    try:
        raw = yf.download(ticker, start=str(start), end=str(end),
                          interval="1d", progress=False)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in raw.columns:
            raw[col] = np.nan
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def _build_feature_row(index: str) -> tuple:
    """
    Returns (feature_row dict, current_price float, latest_date).
    Mirrors the Page 2 Live Prediction pipeline exactly.
    """
    ticker = TICKER_MAP[index]
    ohlcv = _fetch_ohlcv(ticker)
    if ohlcv.empty or len(ohlcv) < 20:
        raise RuntimeError(
            f"Insufficient OHLCV data from yfinance ({len(ohlcv)} rows). "
            "Need ≥20 rows for indicators."
        )

    df_ind = compute_features(ohlcv.copy())

    IND_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA", "EMA", "RSI", "MACD", "MACD_Signal",
        "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
    ]
    last = df_ind.iloc[-1]
    feature_row = {}
    bad = []
    for col in IND_COLS:
        v = last.get(col) if hasattr(last, "get") else last[col]
        if pd.isna(v):
            bad.append(col)
        else:
            feature_row[col] = float(v)
    if bad:
        raise RuntimeError(f"NaN in latest indicator row for: {bad}")

    # Forward-fill macro/sentiment from engineered CSV (last non-null row)
    eng = load_engineered_features(index)
    eng_s = eng.sort_values("Date")
    for col in ["FII_Net", "DII_Net", "Repo_Rate", "Reddit_Sentiment"]:
        valid = eng_s[eng_s[col].notna() & (eng_s[col] != 0.0)]
        feature_row[col] = float(valid[col].iloc[-1]) if not valid.empty else 0.0

    # NaN guard
    nans = [f for f in FEATURE_COLS if np.isnan(feature_row.get(f, float("nan")))]
    if nans:
        raise RuntimeError(f"Features still NaN after all steps: {nans}")

    current_price = feature_row["Close"]
    latest_date   = last["Date"].date() if hasattr(last.get("Date", None), "date") else "latest"
    return feature_row, current_price, latest_date


def _run_all_predictions(index: str) -> dict:
    """
    Runs ensemble h=1, h=5 and Prophet T+30 for the given index.
    Returns a dict with all prediction results and ranges.
    """
    feature_row, current_price, latest_date = _build_feature_row(index)

    # Ensemble h=1
    res1 = predict_ensemble(index, 1, feature_row)
    mae1 = MAE_LOOKUP.get((index, 1))

    # Ensemble h=5
    res5 = predict_ensemble(index, 5, feature_row)
    mae5 = MAE_LOOKUP.get((index, 5))

    # Prophet T+30 — staleness-aware (inline, no external helper)
    import datetime as _dt_mod
    _fc_for_stale = load_prophet_forecast(index)   # cached — no extra disk read
    _last_fc_date = _fc_for_stale["ds"].max().date()
    _gap_days     = (_dt_mod.date.today() - _last_fc_date).days
    _staleness    = {"last_date": _last_fc_date, "gap_days": _gap_days, "is_stale": _gap_days > 90}

    if _staleness["is_stale"]:
        # Stale: use the last row of the pre-computed forecast CSV (cached — no disk re-read)
        _fc_csv = load_prophet_forecast(index)
        _fc_row = _fc_csv.iloc[-1]
        prophet_pred  = float(_fc_row["yhat"])
        prophet_lower = float(_fc_row["yhat_lower"])
        prophet_upper = float(_fc_row["yhat_upper"])
        prophet_stale = True
    else:
        # Fresh: live re-run from saved model is valid
        eng      = load_engineered_features(index)
        eng_s    = eng.sort_values("Date")
        last_eng = eng_s.iloc[-1]
        regressors = {
            k: float(last_eng.get(k, 0) or 0)
            for k in ["Repo_Rate", "FII_Net", "DII_Net", "Reddit_Sentiment"]
        }
        _fc_live  = predict_prophet_live(index, regressors)
        _fc_row   = _fc_live.iloc[-1]
        prophet_pred  = float(_fc_row["yhat"])
        prophet_lower = float(_fc_row["yhat_lower"])
        prophet_upper = float(_fc_row["yhat_upper"])
        prophet_stale = False

    def _range(pred, mae):
        if mae:
            return pred - mae, pred + mae
        return pred, pred

    nd_pred  = res1["predicted_price"]
    nd_low, nd_high = _range(nd_pred, mae1)

    nw_pred  = res5["predicted_price"]
    nw_low, nw_high = _range(nw_pred, mae5)

    return {
        "current_price": current_price,
        "latest_date":   latest_date,
        "nd": {"pred": nd_pred, "low": nd_low, "high": nd_high,
               "ret": res1["predicted_return_pct"], "mae": mae1},
        "nw": {"pred": nw_pred, "low": nw_low, "high": nw_high,
               "ret": res5["predicted_return_pct"], "mae": mae5},
        "nm": {"pred": prophet_pred, "low": prophet_lower, "high": prophet_upper,
               "stale": prophet_stale, "as_of_date": str(_last_fc_date),
               "gap_days": _gap_days},
        "index": index.upper(),
    }


# ── Session state for prediction results ──────────────────────────────────
# NOTE: "home_pred_index" is owned by the st.selectbox widget above — never
# write to it manually.  We use a separate key "home_last_pred_index" to
# record which index the stored results belong to.

if "home_pred_results" not in st.session_state:
    st.session_state.home_pred_results    = None
    st.session_state.home_last_pred_index = None

if generate_btn:
    with st.spinner(f"Running inference for {pred_index_label}…"):
        try:
            results = _run_all_predictions(pred_index)
            st.session_state.home_pred_results    = results
            st.session_state.home_last_pred_index = pred_index_label
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.home_pred_results    = None

# ── Display results ────────────────────────────────────────────────────────

R = st.session_state.home_pred_results
if R:
    idx_label = st.session_state.home_last_pred_index

    st.success(f"{idx_label} Predictions Generated!")
    st.info(f"Current {idx_label} Price: ₹{R['current_price']:,.2f}  "
            f"(as of {R['latest_date']})")

    # ── PREDICTION SUMMARY ──────────────────────────────────────────────
    st.markdown("#### Prediction Summary")

    c1, c2, c3 = st.columns(3)

    nd, nw, nm = R["nd"], R["nw"], R["nm"]

    def _has_range(d):
        return abs(d["high"] - d["low"]) > 1e-3

    with c1:
        st.metric("Next Day (T+1)", f"₹{nd['pred']:,.2f}",
                  delta=f"{nd['ret']:+.2f}%")
        if _has_range(nd):
            st.caption(f"Range: ₹{nd['low']:,.0f} – ₹{nd['high']:,.0f}  "
                       f"(± MAE {nd['mae']:,.0f})")
        else:
            st.caption("Single-point estimate (MAE unavailable)")

    with c2:
        st.metric("Next Week (T+5)", f"₹{nw['pred']:,.2f}",
                  delta=f"{nw['ret']:+.2f}%")
        if _has_range(nw):
            st.caption(f"Range: ₹{nw['low']:,.0f} – ₹{nw['high']:,.0f}  "
                       f"(± MAE {nw['mae']:,.0f})")
        else:
            st.caption("Single-point estimate (MAE unavailable)")

    with c3:
        st.metric("Next Month (T+30)", f"₹{nm['pred']:,.2f}")
        st.caption(
            f"Range: ₹{nm['low']:,.0f} – ₹{nm['high']:,.0f}  (Prophet 95% CI)"
        )
        if nm.get("stale"):
            st.warning(
                f"Forecast generated from data as of {nm['as_of_date']} "
                f"({nm['gap_days']} days ago) — Prophet trend becomes unreliable "
                "when extrapolated far past training data. Will be refreshed once "
                "Prophet is refit on recent data.",
                icon="⚠️",
            )

    # ── PREDICTION RANGES (display-only sliders) ────────────────────────
    st.markdown("#### Prediction Ranges")

    _cur = R["current_price"]

    def _slider_bounds(low, high, cur):
        lo = min(low, cur) * 0.998
        hi = max(high, cur) * 1.002
        return lo, hi

    _lo1, _hi1 = _slider_bounds(nd["low"], nd["high"], _cur)
    st.slider(
        f"Next Day — ₹{nd['low']:,.0f} to ₹{nd['high']:,.0f}",
        min_value=float(_lo1), max_value=float(_hi1),
        value=(float(nd["low"]), float(nd["high"])),
        disabled=True, format="₹%.0f",
    )

    _lo5, _hi5 = _slider_bounds(nw["low"], nw["high"], _cur)
    st.slider(
        f"Next Week — ₹{nw['low']:,.0f} to ₹{nw['high']:,.0f}",
        min_value=float(_lo5), max_value=float(_hi5),
        value=(float(nw["low"]), float(nw["high"])),
        disabled=True, format="₹%.0f",
    )

    _lo30, _hi30 = _slider_bounds(nm["low"], nm["high"], _cur)
    st.slider(
        f"Next Month — ₹{nm['low']:,.0f} to ₹{nm['high']:,.0f}  (Prophet CI)",
        min_value=float(_lo30), max_value=float(_hi30),
        value=(float(nm["low"]), float(nm["high"])),
        disabled=True, format="₹%.0f",
    )

    # ── PREDICTION COMPARISON CHARTS ────────────────────────────────────
    st.markdown("#### Prediction Comparison")

    categories = ["Current Price", "Next Day", "Next Week", "Next Month"]
    lows  = [_cur, nd["low"],  nw["low"],  nm["low"]]
    highs = [_cur, nd["high"], nw["high"], nm["high"]]
    preds = [_cur, nd["pred"], nw["pred"], nm["pred"]]

    COLORS = {
        "Current": "#636EFA",
        "Low":     "#EF553B",
        "High":    "#00CC96",
        "Pred":    "#FFA15A",
    }

    def _make_chart(title, bar_vals, bar_label, bar_color):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=[_cur] * 4,
            name="Current Price",
            marker_color=COLORS["Current"],
            opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            x=categories,
            y=bar_vals,
            name=bar_label,
            marker_color=bar_color,
        ))
        fig.add_trace(go.Scatter(
            x=categories,
            y=preds,
            name="Point Forecast",
            mode="markers+lines",
            marker=dict(size=9, color=COLORS["Pred"], symbol="diamond"),
            line=dict(dash="dot", color=COLORS["Pred"], width=1.5),
        ))
        fig.update_layout(
            title=title,
            yaxis_title="Price (INR)",
            barmode="overlay",
            height=340,
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(t=60, b=40),
            yaxis=dict(tickformat="₹,.0f"),
        )
        return fig

    _ch1, _ch2 = st.columns(2)
    with _ch1:
        st.plotly_chart(
            _make_chart("Low Predictions", lows, "Low Estimate", COLORS["Low"]),
            use_container_width=True,
        )
    with _ch2:
        st.plotly_chart(
            _make_chart("High Predictions", highs, "High Estimate", COLORS["High"]),
            use_container_width=True,
        )

    # ── Disclaimer caption ───────────────────────────────────────────────
    st.caption(
        "Next Day / Next Week use live-fetched indicators (same pipeline as Page 2 — "
        "Live Prediction). Macro and sentiment features (FII/DII, Repo Rate, Reddit Sentiment) "
        "are forward-filled from the last known historical value and may be days or weeks stale. "
        "These predictions are **not validated against the paper's reported metrics** and are for "
        "illustrative purposes only."
    )
