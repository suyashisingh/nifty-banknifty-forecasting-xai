import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RENAME_MAP = {
    'trend_sma_fast':      'SMA',
    'trend_ema_fast':      'EMA',
    'momentum_rsi':        'RSI',
    'trend_macd':          'MACD',
    'trend_macd_signal':   'MACD_Signal',
    'volatility_bbh':      'BB_Upper',
    'volatility_bbm':      'BB_Middle',
    'volatility_bbl':      'BB_Lower',
    'volatility_atr':      'ATR',
    'Close_lag1':          'Close_Lag_1',
    'Close_lag2':          'Close_Lag_2',
    'Close_lag3':          'Close_Lag_3',
    'Close_lag5':          'Close_Lag_5',
    'Close_lag10':         'Close_Lag_10',
    'Repo Rate':           'Repo_Rate',
    'sentiment_daily_x':   'Reddit_Sentiment',
}

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'trend_sma_fast', 'trend_ema_fast',
    'momentum_rsi',
    'trend_macd', 'trend_macd_signal',
    'volatility_bbh', 'volatility_bbm', 'volatility_bbl',
    'volatility_atr',
    'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'Close_lag10',
    'FII_Net', 'DII_Net',
    'Repo Rate',
    'sentiment_daily_x',
]


def process_index(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['Date'])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print("WARNING: missing columns in %s: %s" % (os.path.basename(input_path), missing))

    present = [c for c in FEATURE_COLS if c in df.columns]
    df = df[['Date'] + present].copy()
    df = df.rename(columns=RENAME_MAP)

    renamed_feature_cols = [RENAME_MAP.get(c, c) for c in present]
    df = df.dropna(subset=renamed_feature_cols, how='all').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved %s  ->  %d rows x %d cols  ->  %s" % (
        os.path.basename(input_path),
        len(df),
        len(df.columns),
        os.path.basename(output_path),
    ))


if __name__ == '__main__':
    process_index(
        input_path=os.path.join(BASE_DIR, 'data', 'processed', 'nifty_merged_preprocessed.csv'),
        output_path=os.path.join(BASE_DIR, 'data', 'processed', 'nifty_engineered_features.csv'),
    )
    process_index(
        input_path=os.path.join(BASE_DIR, 'data', 'processed', 'banknifty_merged_preprocessed.csv'),
        output_path=os.path.join(BASE_DIR, 'data', 'processed', 'banknifty_engineered_features.csv'),
    )
