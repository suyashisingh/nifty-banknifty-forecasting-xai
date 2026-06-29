"""
Final consolidated evaluation script.
Computes all metrics for all models, horizons, and indices from saved artifacts.
OUTPUT: models/final_evaluation_report.csv
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    import lightgbm, catboost, xgboost, sklearn
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_CSV = os.path.join(MODELS_DIR, 'final_evaluation_report.csv')

# ── Part 1 config ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
HORIZONS    = [1, 5]
INDICES     = ['nifty', 'banknifty']

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_10',
    'FII_Net', 'DII_Net', 'Repo_Rate', 'Reddit_Sentiment',
]

# ── Part 2 config ──────────────────────────────────────────────────────────────
PROPHET_CONFIGURATIONS = [
    {
        'index':       'nifty',
        'data_file':   'nifty_merged_preprocessed.csv',
        'model_file':  'nifty_prophet.pkl',
        'split_ratio': 0.80,
        'split_label': '80/20',
    },
    {
        'index':       'banknifty',
        'data_file':   'banknifty_merged_preprocessed.csv',
        'model_file':  'banknifty_prophet.pkl',
        'split_ratio': 0.90,
        'split_label': '90/10',
    },
]
REGRESSORS = ['Repo_Rate', 'FII_Net', 'DII_Net', 'Reddit_Sentiment']
EVAL_DAYS  = 30


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics_price(actual_price, pred_price, y_test, pred_return):
    """RMSE/MAE/MAPE in price space; DA in return-sign space."""
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae  = mean_absolute_error(actual_price, pred_price)
    mask = actual_price != 0
    mape = np.mean(np.abs((actual_price[mask] - pred_price[mask]) / actual_price[mask])) * 100
    da   = np.mean(np.sign(y_test) == np.sign(pred_return)) * 100
    return rmse, mae, mape, da


def compute_metrics_prophet(actuals, predictions):
    """RMSE/MAE/MAPE in price space; DA from day-over-day sign changes."""
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae  = mean_absolute_error(actuals, predictions)
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    actual_diff = np.diff(actuals)
    pred_diff   = np.diff(predictions)
    da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100 if len(actual_diff) > 0 else np.nan
    residuals   = actuals - predictions
    res_mean    = residuals.mean()
    res_std     = residuals.std()
    res_rmse    = np.sqrt(np.mean(residuals ** 2))
    return rmse, mae, mape, da, res_mean, res_std, res_rmse


# ── Part 1: Stacked Ensemble + Baselines ──────────────────────────────────────

def evaluate_short_horizon(index, h):
    print(f"\n{'=' * 60}")
    print(f"  [Part 1] {index.upper()}  h={h}")
    print(f"{'=' * 60}")

    feat_path = os.path.join(MODELS_DIR, f'{index}_feature_names.txt')
    with open(feat_path) as f:
        _ = [line.strip() for line in f if line.strip()]   # loaded for reference

    scaler = joblib.load(os.path.join(MODELS_DIR, f'{index}_scaler_h{h}.pkl'))
    xgb    = joblib.load(os.path.join(MODELS_DIR, f'{index}_xgb_h{h}.pkl'))
    lgbm   = joblib.load(os.path.join(MODELS_DIR, f'{index}_lgbm_h{h}.pkl'))
    cat    = joblib.load(os.path.join(MODELS_DIR, f'{index}_catboost_h{h}.pkl'))
    ada    = joblib.load(os.path.join(MODELS_DIR, f'{index}_adaboost_h{h}.pkl'))
    meta   = joblib.load(os.path.join(MODELS_DIR, f'{index}_meta_h{h}.pkl'))

    data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{index}_engineered_features.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df_feat = df[FEATURE_COLS].copy()
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    df_feat = df_feat.fillna(df_feat.mean())

    target_col = (df['Close'].shift(-h) - df['Close']) / df['Close'] * 100
    df_feat    = df_feat.iloc[:-h].reset_index(drop=True)
    target_col = target_col.iloc[:-h].reset_index(drop=True)
    close_raw  = df['Close'].iloc[:-h].reset_index(drop=True)

    X         = df_feat.values
    y         = target_col.values
    Close_all = close_raw.values

    split_idx  = int(TRAIN_RATIO * len(X))
    X_test     = X[split_idx:]
    y_test     = y[split_idx:]
    Close_test = Close_all[split_idx:]
    n_test     = len(X_test)

    X_test_s = scaler.transform(X_test)
    print(f"  Test samples: {n_test}")

    p1 = xgb.predict(X_test_s)
    p2 = lgbm.predict(X_test_s)
    p3 = cat.predict(X_test_s)
    p4 = ada.predict(X_test_s)
    ensemble_pred = meta.predict(np.column_stack([p1, p2, p3, p4]))
    xgb_pred      = p1.copy()
    naive_pred    = np.zeros(len(y_test))

    actual_price     = Close_test * (1 + y_test        / 100)
    pred_price_ens   = Close_test * (1 + ensemble_pred / 100)
    pred_price_xgb   = Close_test * (1 + xgb_pred      / 100)
    pred_price_naive = Close_test.copy()

    ens_rmse,   ens_mae,   ens_mape,   ens_da   = compute_metrics_price(actual_price, pred_price_ens,   y_test, ensemble_pred)
    xgb_rmse,   xgb_mae,   xgb_mape,   xgb_da   = compute_metrics_price(actual_price, pred_price_xgb,   y_test, xgb_pred)
    naive_rmse, naive_mae, naive_mape, naive_da  = compute_metrics_price(actual_price, pred_price_naive, y_test, naive_pred)

    print(f"  Ensemble — RMSE={ens_rmse:.4f}  MAE={ens_mae:.4f}  MAPE={ens_mape:.4f}%  DA={ens_da:.2f}%")
    print(f"  XGBoost  — RMSE={xgb_rmse:.4f}  MAE={xgb_mae:.4f}  MAPE={xgb_mape:.4f}%  DA={xgb_da:.2f}%")
    print(f"  Naive    — RMSE={naive_rmse:.4f}  MAE={naive_mae:.4f}  MAPE={naive_mape:.4f}%  DA={naive_da:.2f}%")

    def row(model, rmse, mae, mape, da):
        return {
            'Index':            index.upper(),
            'Model':            model,
            'Horizon':          f'h={h}',
            'Split':            '80/20',
            'N_Test':           n_test,
            'RMSE':             round(rmse,  4),
            'MAE':              round(mae,   4),
            'MAPE':             round(mape,  4),
            'DA':               round(da,    2),
            'Residual_Mean':    np.nan,
            'Residual_Std':     np.nan,
            'Residual_RMSE':    np.nan,
            'Eval_Window_Start': np.nan,
            'Eval_Window_End':   np.nan,
        }

    return [
        row('Ensemble', ens_rmse,   ens_mae,   ens_mape,   ens_da),
        row('XGBoost',  xgb_rmse,   xgb_mae,   xgb_mape,   xgb_da),
        row('Naive',    naive_rmse, naive_mae, naive_mape, naive_da),
    ]


# ── Part 2: Prophet ────────────────────────────────────────────────────────────

def evaluate_prophet(cfg):
    print(f"\n{'=' * 60}")
    print(f"  [Part 2] {cfg['index'].upper()}  Prophet")
    print(f"{'=' * 60}")

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

    model_path = os.path.join(MODELS_DIR, cfg['model_file'])
    model = joblib.load(model_path)

    split_idx   = int(cfg['split_ratio'] * len(df))
    eval_df     = df.iloc[split_idx: split_idx + EVAL_DAYS].reset_index(drop=True)
    eval_future = eval_df[['ds'] + REGRESSORS].copy()
    forecast    = model.predict(eval_future)
    predictions = forecast['yhat'].values
    actuals     = eval_df['y'].values

    rmse, mae, mape, da, res_mean, res_std, res_rmse = compute_metrics_prophet(actuals, predictions)
    win_start = eval_df['ds'].min().date()
    win_end   = eval_df['ds'].max().date()

    print(f"  Eval window: {win_start} to {win_end}  ({len(eval_df)} rows)")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.4f}%  DA={da:.2f}%")
    print(f"  Residual mean={res_mean:.4f}  std={res_std:.4f}  RMSE={res_rmse:.4f}")

    return {
        'Index':             cfg['index'].upper(),
        'Model':             'Prophet',
        'Horizon':           'Next-month',
        'Split':             cfg['split_label'],
        'N_Test':            len(eval_df),
        'RMSE':              round(rmse,     4),
        'MAE':               round(mae,      4),
        'MAPE':              round(mape,     4),
        'DA':                round(da,       2),
        'Residual_Mean':     round(res_mean, 4),
        'Residual_Std':      round(res_std,  4),
        'Residual_RMSE':     round(res_rmse, 4),
        'Eval_Window_Start': str(win_start),
        'Eval_Window_End':   str(win_end),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_rows = []

    # Part 1
    for index in INDICES:
        for h in HORIZONS:
            rows = evaluate_short_horizon(index, h)
            all_rows.extend(rows)

    # Part 2
    for cfg in PROPHET_CONFIGURATIONS:
        row = evaluate_prophet(cfg)
        all_rows.append(row)

    COLUMNS = [
        'Index', 'Model', 'Horizon', 'Split', 'N_Test',
        'RMSE', 'MAE', 'MAPE', 'DA',
        'Residual_Mean', 'Residual_Std', 'Residual_RMSE',
        'Eval_Window_Start', 'Eval_Window_End',
    ]

    report = pd.DataFrame(all_rows, columns=COLUMNS)
    report.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(f"\n{'=' * 130}")
    print("FINAL EVALUATION REPORT")
    print(f"{'=' * 130}")
    print(report.to_string(index=False))
    print(f"{'=' * 130}")
