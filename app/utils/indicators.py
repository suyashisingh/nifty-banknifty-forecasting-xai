"""
Technical indicator computation matching the training pipeline exactly.

Training pipeline (from source files):
  data_preprocessing.py  -> add_all_ta_features(fillna=True) + add_lag_features(Close, lags=[1,2,3,5,10])
  feature_engineering.py -> RENAME_MAP renames ta column names to the 9 indicator feature names

The ta library column names produced by add_all_ta_features that we use:
  trend_sma_fast    -> SMA
  trend_ema_fast    -> EMA
  momentum_rsi      -> RSI
  trend_macd        -> MACD
  trend_macd_signal -> MACD_Signal
  volatility_bbh    -> BB_Upper
  volatility_bbm    -> BB_Middle
  volatility_bbl    -> BB_Lower
  volatility_atr    -> ATR
  Close_lag1/2/3/5/10 -> Close_Lag_1/2/3/5/10
"""

import pandas as pd

# Mirrors feature_engineering.py RENAME_MAP exactly
RENAME_MAP = {
    "trend_sma_fast":    "SMA",
    "trend_ema_fast":    "EMA",
    "momentum_rsi":      "RSI",
    "trend_macd":        "MACD",
    "trend_macd_signal": "MACD_Signal",
    "volatility_bbh":    "BB_Upper",
    "volatility_bbm":    "BB_Middle",
    "volatility_bbl":    "BB_Lower",
    "volatility_atr":    "ATR",
    "Close_lag1":        "Close_Lag_1",
    "Close_lag2":        "Close_Lag_2",
    "Close_lag3":        "Close_Lag_3",
    "Close_lag5":        "Close_Lag_5",
    "Close_lag10":       "Close_Lag_10",
}


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with at minimum Open/High/Low/Close columns (and optionally Volume),
    compute all technical indicator and lag features matching the training pipeline.

    Input df must have enough rows for the largest window (20-period SMA/BB needs 20 rows,
    lag-10 needs 10 rows — recommend at least 30 rows for reliable output).

    Returns the same DataFrame with all indicator and lag columns added and renamed
    to match the 23-feature schema expected by the scaler/models.
    """
    from ta import add_all_ta_features  # imported here to avoid hard dep at module load

    df = df.copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Step 1: compute all ta features — matches data_preprocessing.add_technical_indicators
    df = add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )

    # Step 2: add Close lags — matches data_preprocessing.add_lag_features(df, ['Close'], lags=[1,2,3,5,10])
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)

    # Step 3: rename to final feature names — matches feature_engineering.py RENAME_MAP
    df = df.rename(columns=RENAME_MAP)

    return df


def get_last_feature_row(df: pd.DataFrame) -> dict:
    """
    Return the last row of a compute_features() output as a feature dict.
    Fills any remaining NaN lag values with 0 to match training ffill/fillna(0) behavior.
    """
    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA", "EMA", "RSI", "MACD", "MACD_Signal",
        "BB_Upper", "BB_Middle", "BB_Lower", "ATR",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
    ]
    row = df.iloc[-1]
    return {col: float(row[col]) if pd.notna(row.get(col)) else 0.0 for col in FEATURE_COLS}
