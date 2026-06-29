import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

try:
    import lightgbm, catboost, xgboost, sklearn
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_PNG = os.path.join(MODELS_DIR, 'fig3_predicted_vs_actual.png')

TRAIN_RATIO  = 0.80
CONFIGURATIONS = [
    ('nifty',     1, 'NIFTY — Next-Day'),
    ('nifty',     5, 'NIFTY — Next-Week'),
    ('banknifty', 1, 'BANKNIFTY — Next-Day'),
    ('banknifty', 5, 'BANKNIFTY — Next-Week'),
]

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_10',
    'FII_Net', 'DII_Net', 'Repo_Rate', 'Reddit_Sentiment',
]


def load_and_predict(index, h):
    feat_path = os.path.join(MODELS_DIR, f'{index}_feature_names.txt')
    with open(feat_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]

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

    X_test_s = scaler.transform(X_test)

    p1 = xgb.predict(X_test_s)
    p2 = lgbm.predict(X_test_s)
    p3 = cat.predict(X_test_s)
    p4 = ada.predict(X_test_s)
    ensemble_pred = meta.predict(np.column_stack([p1, p2, p3, p4]))

    pred_price   = Close_test * (1 + ensemble_pred / 100)
    actual_price = Close_test * (1 + y_test        / 100)

    print(f"  {index.upper()} h={h}: {len(X_test)} test samples")
    return actual_price, pred_price


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes_flat = axes.flatten()

actual_handle = None
pred_handle   = None

for ax, (index, h, title) in zip(axes_flat, CONFIGURATIONS):
    actual_price, pred_price = load_and_predict(index, h)
    idx = np.arange(len(actual_price))

    a_line, = ax.plot(idx, actual_price, color='#1565C0', linewidth=1.2,
                      linestyle='-',  label='Actual')
    p_line, = ax.plot(idx, pred_price,  color='#E53935', linewidth=1.0,
                      linestyle='--', label='Predicted')

    if actual_handle is None:
        actual_handle = a_line
        pred_handle   = p_line

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Sample Index', fontsize=10)
    ax.set_ylabel('Price (index points)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Stacked Ensemble: Predicted vs Actual Prices', fontsize=14,
             fontweight='bold', y=1.01)
fig.legend([actual_handle, pred_handle], ['Actual', 'Predicted'],
           loc='upper center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 1.0), frameon=False)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PNG}")
