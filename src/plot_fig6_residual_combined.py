import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    from statsmodels.graphics.tsaplots import plot_acf
    from sklearn.metrics import mean_squared_error
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_PNG = os.path.join(MODELS_DIR, 'fig6_residual_combined.png')

TRAIN_RATIO = 0.80
EVAL_DAYS   = 30

CONFIGURATIONS = [
    {
        'index':       'nifty',
        'data_file':   'nifty_merged_preprocessed.csv',
        'model_file':  'nifty_prophet.pkl',
        'split_ratio': 0.80,
        'title':       'NIFTY',
    },
    {
        'index':       'banknifty',
        'data_file':   'banknifty_merged_preprocessed.csv',
        'model_file':  'banknifty_prophet.pkl',
        'split_ratio': 0.90,
        'title':       'BANKNIFTY',
    },
]

REGRESSORS = ['Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment']


def compute_residuals(cfg):
    # STEP 1 — Load data
    data_path = os.path.join(BASE_DIR, 'data', 'processed', cfg['data_file'])
    raw = pd.read_csv(data_path)
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if 'Repo Rate' in raw.columns and 'Repo_Rate' not in raw.columns:
        raw = raw.rename(columns={'Repo Rate': 'Repo_Rate'})
    if 'sentiment_daily_x' in raw.columns and 'Reddit_Sentiment' not in raw.columns:
        raw = raw.rename(columns={'sentiment_daily_x': 'Reddit_Sentiment'})

    df = raw[['Date', 'Close'] + REGRESSORS].rename(columns={'Date': 'ds', 'Close': 'y'})
    for col in REGRESSORS:
        df[col] = df[col].ffill().fillna(0)

    # Load model
    model_path = os.path.join(MODELS_DIR, cfg['model_file'])
    model = joblib.load(model_path)

    # STEP 2 — Eval predictions
    split_idx   = int(cfg['split_ratio'] * len(df))
    eval_df     = df.iloc[split_idx: split_idx + EVAL_DAYS].reset_index(drop=True)
    eval_future = eval_df[['ds'] + REGRESSORS].copy()
    forecast    = model.predict(eval_future)
    predictions = forecast['yhat'].values
    actuals     = eval_df['y'].values
    residuals   = actuals - predictions

    # Print summary
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"  {cfg['title']:<12}  eval window: {eval_df['ds'].min().date()} to "
          f"{eval_df['ds'].max().date()}  ({len(eval_df)} rows)")
    print(f"  {'':12}  mean={residuals.mean():.4f}  std={residuals.std():.4f}  "
          f"RMSE={rmse:.4f}")

    return eval_df['ds'], residuals


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Prophet Residual Diagnostics — NIFTY and BANKNIFTY',
             fontsize=14, fontweight='bold')

for row_idx, cfg in enumerate(CONFIGURATIONS):
    print(f"\n{'=' * 60}")
    print(f"  {cfg['title']}")
    print(f"{'=' * 60}")

    dates, residuals = compute_residuals(cfg)
    ax_time, ax_hist, ax_acf = axes[row_idx]

    # Column 1 — Residuals over time
    ax_time.plot(dates, residuals, marker='o', linewidth=1.1, markersize=3,
                 color='steelblue')
    ax_time.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_time.set_title(f"{cfg['title']} — Residuals Over Time", fontsize=11, fontweight='bold')
    ax_time.set_xlabel('Date', fontsize=9)
    ax_time.set_ylabel('Residual (Actual - Predicted)', fontsize=9)
    ax_time.tick_params(axis='x', rotation=30)
    ax_time.grid(linestyle='--', alpha=0.35)
    ax_time.spines[['top', 'right']].set_visible(False)

    # Column 2 — Residual distribution
    ax_hist.hist(residuals, bins=10, color='steelblue', edgecolor='white', alpha=0.85)
    ax_hist.axvline(residuals.mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean = {residuals.mean():.2f}')
    ax_hist.set_title(f"{cfg['title']} — Residual Distribution", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel('Residual', fontsize=9)
    ax_hist.set_ylabel('Frequency', fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(linestyle='--', alpha=0.35)
    ax_hist.spines[['top', 'right']].set_visible(False)

    # Column 3 — ACF
    lags = min(15, len(residuals) - 1)
    plot_acf(residuals, lags=lags, ax=ax_acf, alpha=0.05, zero=False)
    ax_acf.set_title(f"{cfg['title']} — Residual ACF", fontsize=11, fontweight='bold')
    ax_acf.set_xlabel('Lag', fontsize=9)
    ax_acf.set_ylabel('Autocorrelation', fontsize=9)
    ax_acf.grid(linestyle='--', alpha=0.35)
    ax_acf.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PNG}")
