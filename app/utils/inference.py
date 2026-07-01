"""
Ensemble and Prophet inference utilities.

Ensemble pipeline:
  X_raw (1,23) -> scaler.transform -> [xgb,lgbm,catboost,adaboost].predict
               -> stack (1,4) -> meta.predict -> % return
               -> predicted_price = base_close * (1 + return/100)

Prophet pipeline:
  Build future DataFrame with ds + 4 regressors -> model.predict -> yhat/bounds
"""

import numpy as np
import pandas as pd

from utils.data_loader import load_ensemble_artifacts, load_prophet_model

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA", "EMA", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
    "FII_Net", "DII_Net", "Repo_Rate", "Reddit_Sentiment",
]

PROPHET_REGRESSORS = ["Repo_Rate", "FII_Net", "DII_Net", "Reddit_Sentiment"]


def validate_feature_row(feature_row: dict) -> None:
    """Raise ValueError listing any missing or None/NaN features."""
    bad = []
    for f in FEATURE_COLS:
        val = feature_row.get(f)
        if val is None:
            bad.append(f"{f} (missing)")
        elif isinstance(val, float) and np.isnan(val):
            bad.append(f"{f} (NaN)")
    if bad:
        raise ValueError(
            f"Cannot run inference — the following features are invalid:\n"
            + "\n".join(f"  • {b}" for b in bad)
        )


def predict_ensemble(index: str, horizon: int, feature_row: dict) -> dict:
    """
    Run stacked ensemble inference for one sample.

    Parameters
    ----------
    index       : 'nifty' or 'banknifty'
    horizon     : 1 or 5
    feature_row : dict mapping all 23 FEATURE_COLS names to float values

    Returns
    -------
    dict with keys:
        predicted_return_pct  – meta-learner output (% return over horizon)
        predicted_price       – base_close * (1 + return/100)
        base_close            – the Close value used for back-conversion
        base_model_preds      – dict of {model_name: predicted_return_pct}
    """
    validate_feature_row(feature_row)

    arts = load_ensemble_artifacts(index, horizon)

    X_raw = np.array([[feature_row[f] for f in FEATURE_COLS]], dtype=np.float64)
    X_sc = arts["scaler"].transform(X_raw)

    p_xgb = float(arts["xgb"].predict(X_sc)[0])
    p_lgbm = float(arts["lgbm"].predict(X_sc)[0])
    p_cat = float(arts["catboost"].predict(X_sc)[0])
    p_ada = float(arts["adaboost"].predict(X_sc)[0])

    meta_input = np.array([[p_xgb, p_lgbm, p_cat, p_ada]])
    ret = float(arts["meta"].predict(meta_input)[0])

    base_close = float(feature_row["Close"])
    predicted_price = base_close * (1 + ret / 100)

    return {
        "predicted_return_pct": ret,
        "predicted_price": predicted_price,
        "base_close": base_close,
        "base_model_preds": {
            "XGBoost": p_xgb,
            "LightGBM": p_lgbm,
            "CatBoost": p_cat,
            "AdaBoost": p_ada,
        },
    }


def predict_prophet_live(index: str, regressors: dict) -> pd.DataFrame:
    """
    Re-run Prophet inference for the next 30 business days using the saved model.

    Parameters
    ----------
    index      : 'nifty' or 'banknifty'
    regressors : dict with keys Repo_Rate, FII_Net, DII_Net, Reddit_Sentiment
                 (scalar float values — forward-filled from last known)

    Returns
    -------
    DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    model = load_prophet_model(index)
    last_date = pd.Timestamp.today().normalize()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=30
    )
    future_df = pd.DataFrame({"ds": future_dates})
    for reg, val in regressors.items():
        future_df[reg] = float(val)

    forecast = model.predict(future_df)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
