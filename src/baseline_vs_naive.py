import os
import sys
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm, catboost, xgboost, sklearn
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_RATIO = 0.80
HORIZONS    = [1, 5]
INDICES     = ['nifty', 'banknifty']
OUTPUT_CSV  = os.path.join(MODELS_DIR, 'baseline_vs_naive.csv')

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_10',
    'FII_Net', 'DII_Net', 'Repo_Rate', 'Reddit_Sentiment',
]


def compute_metrics(actual_price, pred_price, y_test, pred_return):
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae  = mean_absolute_error(actual_price, pred_price)
    mask = actual_price != 0
    mape = np.mean(np.abs((actual_price[mask] - pred_price[mask]) / actual_price[mask])) * 100
    da   = np.mean(np.sign(y_test) == np.sign(pred_return)) * 100
    return rmse, mae, mape, da


def evaluate_index_horizon(index, h):
    print(f"\n{'=' * 60}")
    print(f"  {index.upper()}  h={h}")
    print(f"{'=' * 60}")

    # STEP 1 — Load artifacts
    feat_path = os.path.join(MODELS_DIR, f'{index}_feature_names.txt')
    with open(feat_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]

    scaler  = joblib.load(os.path.join(MODELS_DIR, f'{index}_scaler_h{h}.pkl'))
    xgb     = joblib.load(os.path.join(MODELS_DIR, f'{index}_xgb_h{h}.pkl'))
    lgbm    = joblib.load(os.path.join(MODELS_DIR, f'{index}_lgbm_h{h}.pkl'))
    cat     = joblib.load(os.path.join(MODELS_DIR, f'{index}_catboost_h{h}.pkl'))
    ada     = joblib.load(os.path.join(MODELS_DIR, f'{index}_adaboost_h{h}.pkl'))
    meta    = joblib.load(os.path.join(MODELS_DIR, f'{index}_meta_h{h}.pkl'))

    # STEP 2 — Prepare data
    data_path = os.path.join(BASE_DIR, 'data', 'processed', f'{index}_engineered_features.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df_feat = df[FEATURE_COLS].copy()
    df_feat = df_feat.replace([np.inf, -np.inf], 0)
    df_feat = df_feat.fillna(df_feat.mean())

    # Build target in return space, drop last h rows
    target_col = (df['Close'].shift(-h) - df['Close']) / df['Close'] * 100
    df_feat = df_feat.iloc[:-h].reset_index(drop=True)
    target_col = target_col.iloc[:-h].reset_index(drop=True)
    close_raw  = df['Close'].iloc[:-h].reset_index(drop=True)

    X = df_feat.values
    y = target_col.values
    Close_all = close_raw.values

    split_idx  = int(TRAIN_RATIO * len(X))
    X_test     = X[split_idx:]
    y_test     = y[split_idx:]
    Close_test = Close_all[split_idx:]

    X_test_s = scaler.transform(X_test)
    print(f"  Test samples: {len(X_test)}")

    # STEP 3 — Predictions
    p1 = xgb.predict(X_test_s)
    p2 = lgbm.predict(X_test_s)
    p3 = cat.predict(X_test_s)
    p4 = ada.predict(X_test_s)
    meta_input    = np.column_stack([p1, p2, p3, p4])
    ensemble_pred = meta.predict(meta_input)

    xgb_pred   = p1.copy()
    naive_pred  = np.zeros(len(y_test))

    # STEP 4 — Price space
    actual_price       = Close_test * (1 + y_test        / 100)
    pred_price_ens     = Close_test * (1 + ensemble_pred / 100)
    pred_price_xgb     = Close_test * (1 + xgb_pred      / 100)
    pred_price_naive   = Close_test.copy()

    # STEP 5 — Metrics
    ens_rmse,   ens_mae,   ens_mape,   ens_da   = compute_metrics(actual_price, pred_price_ens,   y_test, ensemble_pred)
    xgb_rmse,   xgb_mae,   xgb_mape,   xgb_da   = compute_metrics(actual_price, pred_price_xgb,   y_test, xgb_pred)
    naive_rmse, naive_mae, naive_mape, naive_da  = compute_metrics(actual_price, pred_price_naive, y_test, naive_pred)

    # STEP 6 — Improvement percentages (Ensemble vs baselines)
    vs_xgb_rmse_impr   = (xgb_rmse   - ens_rmse)   / xgb_rmse   * 100
    vs_xgb_mape_impr   = (xgb_mape   - ens_mape)   / xgb_mape   * 100
    vs_naive_rmse_impr = (naive_rmse  - ens_rmse)   / naive_rmse * 100
    vs_naive_mape_impr = (naive_mape  - ens_mape)   / naive_mape * 100

    print(f"  Ensemble  — RMSE={ens_rmse:.4f}  MAE={ens_mae:.4f}  MAPE={ens_mape:.4f}%  DA={ens_da:.2f}%")
    print(f"  XGBoost   — RMSE={xgb_rmse:.4f}  MAE={xgb_mae:.4f}  MAPE={xgb_mape:.4f}%  DA={xgb_da:.2f}%")
    print(f"  Naive     — RMSE={naive_rmse:.4f}  MAE={naive_mae:.4f}  MAPE={naive_mape:.4f}%  DA={naive_da:.2f}%")
    print(f"  Impr vs XGB  — RMSE: {vs_xgb_rmse_impr:.2f}%  MAPE: {vs_xgb_mape_impr:.2f}%")
    print(f"  Impr vs Naive— RMSE: {vs_naive_rmse_impr:.2f}%  MAPE: {vs_naive_mape_impr:.2f}%")

    rows = [
        {
            'Index': index.upper(), 'Model': 'Ensemble', 'Horizon': h,
            'RMSE': round(ens_rmse, 4), 'MAE': round(ens_mae, 4),
            'MAPE': round(ens_mape, 4), 'DA': round(ens_da, 2),
            'vs_XGB_RMSE_impr':   round(vs_xgb_rmse_impr, 2),
            'vs_XGB_MAPE_impr':   round(vs_xgb_mape_impr, 2),
            'vs_Naive_RMSE_impr': round(vs_naive_rmse_impr, 2),
            'vs_Naive_MAPE_impr': round(vs_naive_mape_impr, 2),
        },
        {
            'Index': index.upper(), 'Model': 'XGBoost', 'Horizon': h,
            'RMSE': round(xgb_rmse, 4), 'MAE': round(xgb_mae, 4),
            'MAPE': round(xgb_mape, 4), 'DA': round(xgb_da, 2),
            'vs_XGB_RMSE_impr': np.nan, 'vs_XGB_MAPE_impr': np.nan,
            'vs_Naive_RMSE_impr': np.nan, 'vs_Naive_MAPE_impr': np.nan,
        },
        {
            'Index': index.upper(), 'Model': 'Naive', 'Horizon': h,
            'RMSE': round(naive_rmse, 4), 'MAE': round(naive_mae, 4),
            'MAPE': round(naive_mape, 4), 'DA': round(naive_da, 2),
            'vs_XGB_RMSE_impr': np.nan, 'vs_XGB_MAPE_impr': np.nan,
            'vs_Naive_RMSE_impr': np.nan, 'vs_Naive_MAPE_impr': np.nan,
        },
    ]

    summary = {
        'index': index.upper(), 'h': h,
        'ens_mape': ens_mape, 'xgb_mape': xgb_mape, 'naive_mape': naive_mape,
        'vs_xgb_mape_impr': vs_xgb_mape_impr, 'vs_naive_mape_impr': vs_naive_mape_impr,
    }
    return rows, summary


if __name__ == '__main__':
    all_rows    = []
    all_summary = []

    for index in INDICES:
        for h in HORIZONS:
            rows, summary = evaluate_index_horizon(index, h)
            all_rows.extend(rows)
            all_summary.append(summary)

    # STEP 7 & 8 — DataFrame and save
    results_df = pd.DataFrame(all_rows, columns=[
        'Index', 'Model', 'Horizon', 'RMSE', 'MAE', 'MAPE', 'DA',
        'vs_XGB_RMSE_impr', 'vs_XGB_MAPE_impr',
        'vs_Naive_RMSE_impr', 'vs_Naive_MAPE_impr',
    ])
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results -> {OUTPUT_CSV}")

    # STEP 9 — Full table
    print(f"\n{'=' * 110}")
    print("FULL RESULTS TABLE")
    print(f"{'=' * 110}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.to_string(index=False))

    # STEP 10 — Clean summary
    print(f"\n{'=' * 60}")
    print("BASELINE VS NAIVE — SUMMARY")
    print(f"{'=' * 60}")
    for s in all_summary:
        label = f"{s['index']} h={s['h']}"
        print(
            f"  {label:<16}: Ensemble MAPE {s['ens_mape']:.4f}% | "
            f"XGBoost {s['xgb_mape']:.4f}% | "
            f"Naive {s['naive_mape']:.4f}% | "
            f"Impr vs XGB: {s['vs_xgb_mape_impr']:.2f}% | "
            f"Impr vs Naive: {s['vs_naive_mape_impr']:.2f}%"
        )
    print(f"{'=' * 60}")
