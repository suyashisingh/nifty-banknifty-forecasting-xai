import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib

# Suppress Prophet / cmdstanpy verbose output
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'banknifty_merged_preprocessed.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

REGRESSORS = ['Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment']


def build_holidays():
    """NSE market holidays for 2010-2025 (fixed approximate dates per spec)."""
    templates = [
        ('Republic Day',     1, 26),
        ('Good Friday',      4,  2),
        ('Eid al-Fitr',      4, 21),
        ('Maharashtra Day',  5,  1),
        ('Independence Day', 8, 15),
        ('Janmashtami',      8, 26),
        ('Gandhi Jayanti',  10,  2),
        ('Diwali',          11,  1),
        ('Christmas',       12, 25),
    ]
    rows = []
    for year in range(2010, 2026):
        for name, month, day in templates:
            try:
                rows.append({'ds': pd.Timestamp(year, month, day), 'holiday': name})
            except ValueError:
                pass
    return pd.DataFrame(rows)


def compute_mape(actuals, predictions):
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    mask = actuals != 0
    return np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- STEP 1: Data preparation ---
    raw = pd.read_csv(DATA_PATH)
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True)

    # Normalise column names
    if 'Repo Rate' in raw.columns and 'Repo_Rate' not in raw.columns:
        raw = raw.rename(columns={'Repo Rate': 'Repo_Rate'})
    if 'sentiment_daily_x' in raw.columns and 'Reddit_Sentiment' not in raw.columns:
        raw = raw.rename(columns={'sentiment_daily_x': 'Reddit_Sentiment'})

    needed = ['Date', 'Close', 'Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment']
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        print(f"ERROR: columns not found in preprocessed file: {missing}")
        print(f"Available columns: {list(raw.columns)}")
        sys.exit(1)

    df = raw[needed].copy()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Forward-fill then zero-fill regressor columns
    for col in REGRESSORS:
        df[col] = df[col].ffill().fillna(0)

    split_idx = int(0.90 * len(df))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df  = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows "
          f"({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
    print(f"Test : {len(test_df)} rows "
          f"({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")

    # --- STEP 2: Prophet model configuration ---
    holidays = build_holidays()
    model = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        weekly_seasonality=False,   # added manually below with custom fourier_order
        yearly_seasonality=False,   # added manually below with custom fourier_order
        changepoint_prior_scale=0.15,
        interval_width=0.95,
        holidays=holidays,
    )
    model.add_seasonality(name='weekly', period=7,      fourier_order=5)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    # --- STEP 3: Add exogenous regressors ---
    for reg in REGRESSORS:
        model.add_regressor(reg)

    # --- STEP 4: Training ---
    print("Fitting BANKNIFTY Prophet model...")
    model.fit(train_df[['ds', 'y'] + REGRESSORS])
    print("Fitting complete.")

    # --- STEP 5: Evaluation on first 30 rows of test set ---
    eval_df      = test_df.iloc[:30].copy()
    eval_future  = eval_df[['ds'] + REGRESSORS].copy()
    forecast     = model.predict(eval_future)
    actuals      = eval_df['y'].values
    predictions  = forecast['yhat'].values

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae  = mean_absolute_error(actuals, predictions)
    mape = compute_mape(actuals, predictions)
    da   = np.mean(np.sign(np.diff(actuals)) == np.sign(np.diff(predictions))) * 100

    # --- STEP 6: Future 30-day forecast ---
    last_date    = df['ds'].max()
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=30
    )
    future_df = pd.DataFrame({'ds': future_dates})
    for reg in REGRESSORS:
        future_df[reg] = df[reg].iloc[-1]

    future_forecast = model.predict(future_df)
    forecast_out    = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    forecast_path = os.path.join(MODELS_DIR, 'banknifty_prophet_forecast.csv')
    forecast_out.to_csv(forecast_path, index=False)

    # --- STEP 7: Save model ---
    model_path = os.path.join(MODELS_DIR, 'banknifty_prophet.pkl')
    joblib.dump(model, model_path)

    # --- STEP 8: Final summary ---
    first = forecast_out.iloc[0]
    last  = forecast_out.iloc[-1]

    print(f"\n{'=' * 60}")
    print("BANKNIFTY PROPHET — FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Train period : {train_df['ds'].min().date()} to {train_df['ds'].max().date()} ({len(train_df)} rows)")
    print(f"Test period  : {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} rows)")
    print(f"Regressors   : {', '.join(REGRESSORS)}")
    print(f"{'-' * 60}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MAE   : {mae:.4f}")
    print(f"MAPE  : {mape:.4f}%")
    print(f"DA    : {da:.2f}%")
    print(f"{'-' * 60}")
    print(f"30-day forecast range: {forecast_out['yhat'].min():.2f} to {forecast_out['yhat'].max():.2f}")
    print(f"First predicted date : {first['ds'].date()} | yhat={first['yhat']:.2f} "
          f"[{first['yhat_lower']:.2f}, {first['yhat_upper']:.2f}]")
    print(f"Last predicted date  : {last['ds'].date()} | yhat={last['yhat']:.2f} "
          f"[{last['yhat_lower']:.2f}, {last['yhat_upper']:.2f}]")
    print(f"{'=' * 60}")
    print(f"\nSaved: {model_path}")
    print(f"Saved: {forecast_path}")
