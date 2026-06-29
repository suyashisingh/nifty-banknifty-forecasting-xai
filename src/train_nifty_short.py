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

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'nifty_engineered_features.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_10',
    'FII_Net', 'DII_Net', 'Repo_Rate', 'Reddit_Sentiment',
]


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def make_model(key):
    if key == 'xgb':
        return XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    if key == 'lgbm':
        return LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    if key == 'catboost':
        return CatBoostRegressor(
            iterations=300, depth=5, learning_rate=0.05,
            random_state=42, verbose=0,
        )
    if key == 'adaboost':
        return AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    raise ValueError(f"Unknown model key: {key}")


MODEL_KEYS = ['xgb', 'lgbm', 'catboost', 'adaboost']
MODEL_NAMES = {'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'catboost': 'CatBoost', 'adaboost': 'AdaBoost'}


def train_horizon(h, df_base):
    print(f"\n{'=' * 60}")
    print(f"  Horizon h={h} ({'next day' if h == 1 else 'next week'})")
    print(f"{'=' * 60}")

    df = df_base.copy()
    df[f'Target_{h}'] = (df['Close'].shift(-h) - df['Close']) / df['Close'] * 100
    df = df.iloc[:-h].reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[f'Target_{h}'].values

    split_idx = int(0.8 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    Close_test = df['Close'].values[split_idx:]
    print(f"  Split: {len(X_train)} train / {len(X_test)} test")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    scaler_path = os.path.join(MODELS_DIR, f'nifty_scaler_h{h}.pkl')
    joblib.dump(scaler, scaler_path)

    # Level 1: OOF meta-feature generation
    n_train, n_test = len(X_train_sc), len(X_test_sc)
    meta_train = np.zeros((n_train, len(MODEL_KEYS)))
    meta_test = np.zeros((n_test, len(MODEL_KEYS)))
    tscv = TimeSeriesSplit(n_splits=5)

    saved_model_paths = []

    for mi, key in enumerate(MODEL_KEYS):
        print(f"  [{mi+1}/4] Level-1 OOF: {MODEL_NAMES[key]}")

        for tr_idx, val_idx in tscv.split(X_train_sc):
            fold_model = make_model(key)
            fold_model.fit(X_train_sc[tr_idx], y_train[tr_idx])
            meta_train[val_idx, mi] = fold_model.predict(X_train_sc[val_idx])

        full_model = make_model(key)
        full_model.fit(X_train_sc, y_train)
        meta_test[:, mi] = full_model.predict(X_test_sc)

        model_path = os.path.join(MODELS_DIR, f'nifty_{key}_h{h}.pkl')
        joblib.dump(full_model, model_path)
        saved_model_paths.append(model_path)

    # Level 2: Ridge meta-learner
    print(f"  [5/5] Level-2: Ridge meta-learner")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, y_train)
    y_pred = meta_model.predict(meta_test)

    meta_path = os.path.join(MODELS_DIR, f'nifty_meta_h{h}.pkl')
    joblib.dump(meta_model, meta_path)
    saved_model_paths.append(meta_path)
    saved_model_paths.append(scaler_path)

    # Back-convert predicted returns to prices for interpretable error
    actual_price = Close_test * (1 + y_test / 100)
    pred_price   = Close_test * (1 + y_pred / 100)

    # Return-space metrics
    rmse_ret  = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_ret   = mean_absolute_error(y_test, y_pred)
    mape_ret  = mape(y_test, y_pred)

    # Price-space metrics
    rmse_price = np.sqrt(mean_squared_error(actual_price, pred_price))
    mae_price  = mean_absolute_error(actual_price, pred_price)
    mape_price = mape(actual_price, pred_price)

    print(f"\n  --- Evaluation h={h} ---")
    print(f"  Return metrics (% return space):")
    print(f"    RMSE : {rmse_ret:.6f}%")
    print(f"    MAE  : {mae_ret:.6f}%")
    print(f"    MAPE : {mape_ret:.4f}%")
    print(f"  Price metrics (reconstructed INR):")
    print(f"    RMSE : {rmse_price:.4f}")
    print(f"    MAE  : {mae_price:.4f}")
    print(f"    MAPE : {mape_price:.4f}%")

    # Directional Accuracy — percentage of correctly predicted up/down moves
    da = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
    print(f"  Directional Accuracy:")
    print(f"    DA   : {da:.2f}%")

    print(f"\n  First 5 actual vs predicted:")
    for i in range(5):
        print(f"    [{i}]  return: actual={y_test[i]:.4f}%  pred={y_pred[i]:.4f}%"
              f"  |  price: actual={actual_price[i]:.2f}  pred={pred_price[i]:.2f}")

    print(f"\n  Saved artifacts:")
    for p in saved_model_paths:
        print(f"    {p}")

    print(f"\n[COMPLETE] h={h} | PriceRMSE={rmse_price:.4f} | PriceMAE={mae_price:.4f} | PriceMAPE={mape_price:.4f}% | DA={da:.2f}%")

    return rmse_ret, mae_ret, mape_ret, rmse_price, mae_price, mape_price, da


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} rows | {DATA_PATH}")

    feature_names_path = os.path.join(MODELS_DIR, 'nifty_feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(FEATURE_COLS))
    print(f"Saved feature names -> {feature_names_path}")

    results = {}
    for h in [1, 5]:
        results[h] = train_horizon(h, df)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    for h, (rmse_ret, mae_ret, mape_ret, rmse_price, mae_price, mape_price, da) in results.items():
        label = 'next day ' if h == 1 else 'next week'
        print(f"  h={h} ({label})")
        print(f"    Return : RMSE={rmse_ret:.6f}%  MAE={mae_ret:.6f}%  MAPE={mape_ret:.4f}%")
        print(f"    Price  : RMSE={rmse_price:.4f}  MAE={mae_price:.4f}  MAPE={mape_price:.4f}%")
        print(f"    DA     : {da:.2f}%")
