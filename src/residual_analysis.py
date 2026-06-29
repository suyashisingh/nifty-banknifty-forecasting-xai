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
OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'residual_analysis')

TRAIN_RATIO = 0.80
EVAL_DAYS   = 30
INDICES     = ['nifty', 'banknifty']

CONFIGURATIONS = [
    {
        'index':      'nifty',
        'data_file':  'nifty_merged_preprocessed.csv',
        'model_file': 'nifty_prophet.pkl',
        'regressors': ['Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment'],
    },
    {
        'index':      'banknifty',
        'data_file':  'banknifty_merged_preprocessed.csv',
        'model_file': 'banknifty_prophet.pkl',
        'regressors': ['Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment'],
    },
]


def run_residual_analysis(cfg):
    index      = cfg['index']
    regressors = cfg['regressors']

    print(f"\n{'=' * 60}")
    print(f"  Residual Analysis — {index.upper()}")
    print(f"{'=' * 60}")

    # STEP 1 — Load and prepare data
    data_path = os.path.join(BASE_DIR, 'data', 'processed', cfg['data_file'])
    raw = pd.read_csv(data_path)
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Normalise column names to match what the model was trained with
    if 'Repo Rate' in raw.columns and 'Repo_Rate' not in raw.columns:
        raw = raw.rename(columns={'Repo Rate': 'Repo_Rate'})
    if 'sentiment_daily_x' in raw.columns and 'Reddit_Sentiment' not in raw.columns:
        raw = raw.rename(columns={'sentiment_daily_x': 'Reddit_Sentiment'})

    df = raw[['Date', 'Close'] + regressors].copy()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    for col in regressors:
        df[col] = df[col].ffill().fillna(0)

    split_idx = int(TRAIN_RATIO * len(df))
    eval_df   = df.iloc[split_idx: split_idx + EVAL_DAYS].reset_index(drop=True)
    print(f"  Eval window: {eval_df['ds'].min().date()} to {eval_df['ds'].max().date()} "
          f"({len(eval_df)} rows)")

    # STEP 2 — Load model
    model_path = os.path.join(BASE_DIR, 'models', cfg['model_file'])
    model = joblib.load(model_path)
    print(f"  Loaded model: {cfg['model_file']}")

    # STEP 3 — Predictions and residuals
    eval_future  = eval_df[['ds'] + regressors].copy()
    forecast     = model.predict(eval_future)
    predictions  = forecast['yhat'].values
    actuals      = eval_df['y'].values
    residuals    = actuals - predictions

    # STEP 4 — Plot (1 row × 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{index.upper()} Prophet — Residual Analysis', fontsize=13, fontweight='bold')

    # Plot 1: Residuals over time
    ax1 = axes[0]
    ax1.plot(eval_df['ds'], residuals, marker='o', linewidth=1.2, markersize=3, color='steelblue')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual (Actual - Predicted)')
    ax1.tick_params(axis='x', rotation=30)

    # Plot 2: Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=10, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(residuals.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {residuals.mean():.2f}')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=8)

    # Plot 3: ACF
    ax3 = axes[2]
    lags = min(15, len(residuals) - 1)
    plot_acf(residuals, lags=lags, ax=ax3, alpha=0.05, zero=False)
    ax3.set_title('Residual Autocorrelation (ACF)')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'{index}_prophet_residual_analysis.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {out_path}")

    # STEP 5 — Summary statistics
    mean_res = residuals.mean()
    std_res  = residuals.std()
    min_res  = residuals.min()
    max_res  = residuals.max()
    rmse     = np.sqrt(mean_squared_error(actuals, predictions))
    unbiased = abs(mean_res) < 0.5 * std_res

    print(f"\n  --- Summary Statistics ---")
    print(f"  Mean residual : {mean_res:.4f}")
    print(f"  Std  residual : {std_res:.4f}")
    print(f"  Min  residual : {min_res:.4f}")
    print(f"  Max  residual : {max_res:.4f}")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  Unbiased      : {unbiased}  (|mean| < 0.5 * std)")

    return {
        'Index':        index.upper(),
        'Mean_Residual': round(mean_res, 4),
        'Std_Residual':  round(std_res,  4),
        'Min_Residual':  round(min_res,  4),
        'Max_Residual':  round(max_res,  4),
        'RMSE':          round(rmse,     4),
        'Unbiased':      unbiased,
    }


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    for cfg in CONFIGURATIONS:
        row = run_residual_analysis(cfg)
        summary_rows.append(row)
        print(f"\n[COMPLETE] {cfg['index'].upper()} residual analysis done.")

    # STEP 6 — Save summary CSV
    summary_df  = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'prophet_residual_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    print(f"\n{'=' * 60}")
    print("PROPHET RESIDUAL SUMMARY")
    print(f"{'=' * 60}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary_df.to_string(index=False))
    print(f"{'=' * 60}")
