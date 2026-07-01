"""
Cached loaders for CSV data files and model artifacts.
All paths are resolved relative to pythonProject/ (two levels above this file).
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent.parent.parent   # pythonProject/
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

@st.cache_data
def load_feature_names(index: str) -> list:
    """Return ordered list of 23 feature names from the saved .txt file."""
    path = MODELS_DIR / f"{index}_feature_names.txt"
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_engineered_features(index: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{index}_engineered_features.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_prophet_forecast(index: str) -> pd.DataFrame:
    path = MODELS_DIR / f"{index}_prophet_forecast.csv"
    return pd.read_csv(path, parse_dates=["ds"])


@st.cache_data
def load_baseline_csv() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "baseline_vs_naive.csv")


@st.cache_data
def load_final_evaluation_csv() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "final_evaluation_report.csv")


@st.cache_data
def load_fii_dii() -> pd.DataFrame:
    path = DATA_DIR / "fii_dii_flows_cleaned.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_rbi_repo() -> pd.DataFrame:
    path = DATA_DIR / "rbi_repo_rate_history.csv"
    df = pd.read_csv(path)
    if "Effective Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Effective Date"], format="%d-%b-%y", errors="coerce"
        )
        mask_na = df["Date"].isna()
        if mask_na.any():
            df.loc[mask_na, "Date"] = pd.to_datetime(
                df.loc[mask_na, "Effective Date"], format="%b-%y", errors="coerce"
            )
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["Repo Rate"] = (
            df["Repo Rate"].astype(str).str.replace("%", "").str.strip().astype(float)
        )
    return df


@st.cache_data
def load_reddit_sentiment() -> pd.DataFrame:
    """Return raw reddit sentiment CSV (created_utc is unix timestamp)."""
    path = DATA_DIR / "reddit_nifty_sentiment.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Model artifact loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_ensemble_artifacts(index: str, horizon: int) -> dict:
    """Load all 6 pkl files for one (index, horizon) ensemble configuration."""
    prefix = index
    h = horizon
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "scaler":   joblib.load(MODELS_DIR / f"{prefix}_scaler_h{h}.pkl"),
            "xgb":      joblib.load(MODELS_DIR / f"{prefix}_xgb_h{h}.pkl"),
            "lgbm":     joblib.load(MODELS_DIR / f"{prefix}_lgbm_h{h}.pkl"),
            "catboost": joblib.load(MODELS_DIR / f"{prefix}_catboost_h{h}.pkl"),
            "adaboost": joblib.load(MODELS_DIR / f"{prefix}_adaboost_h{h}.pkl"),
            "meta":     joblib.load(MODELS_DIR / f"{prefix}_meta_h{h}.pkl"),
        }


@st.cache_resource
def load_prophet_model(index: str):
    """Load serialized Prophet model from models/{index}_prophet.pkl."""
    path = MODELS_DIR / f"{index}_prophet.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)
