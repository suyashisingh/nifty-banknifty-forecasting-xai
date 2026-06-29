import os
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Utility Functions ---

def load_csv(path, parse_dates=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=parse_dates, dayfirst=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    else:
        print(f"File not found: {path}")
        return None

def add_technical_indicators(df):
    df = df.copy()
    # Ensure 'Volume' column exists for ta
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return df

def add_lag_features(df, cols, lags=[1, 2, 3, 5, 10]):
    lag_cols = {
        f"{col}_lag{lag}": df[col].shift(lag)
        for col in cols
        for lag in lags
    }
    return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1).copy()

def add_rolling_features(df, cols, windows=[3, 5, 10, 20]):
    for col in cols:
        for win in windows:
            df[f"{col}_rollmean{win}"] = df[col].rolling(win).mean()
            df[f"{col}_rollstd{win}"] = df[col].rolling(win).std()
    df = df.copy()  # Defragment after adding columns
    return df

def preprocess_sentiment(df, date_col, sentiment_col, freq='D'):
    if df is None or date_col not in df.columns or sentiment_col not in df.columns:
        return None
    # Try to parse as seconds since epoch, fallback to direct parsing
    try:
        df[date_col] = pd.to_datetime(df[date_col], unit='s', errors='coerce')
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    daily_sentiment = df.groupby(df[date_col].dt.date)[sentiment_col].mean().reset_index()
    daily_sentiment.columns = ['Date', f'{sentiment_col}_daily']
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'], errors='coerce')
    return daily_sentiment

def merge_on_date(main_df, other_df, how='left'):
    # Ensure both Date columns are datetime64[ns]
    main_df['Date'] = pd.to_datetime(main_df['Date'], errors='coerce')
    other_df['Date'] = pd.to_datetime(other_df['Date'], errors='coerce')
    return pd.merge(main_df, other_df, on='Date', how=how)

def scale_features(df, cols, scaler_type='standard'):
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# --- Load All Raw Data ---

# Market data
nifty = load_csv(os.path.join(BASE_DIR, 'data', 'nifty_raw.csv'), parse_dates=['Date'])
banknifty = load_csv(os.path.join(BASE_DIR, 'data', 'banknifty_raw.csv'), parse_dates=['Date'])
# Macro data
repo = load_csv(os.path.join(BASE_DIR, 'data', 'fii_dii_flows_cleaned.csv'), parse_dates=['Date'])
rbi = load_csv(os.path.join(BASE_DIR, 'data', 'rbi_repo_rate_history.csv'))
if rbi is not None and 'Effective Date' in rbi.columns:
    # Robust date parsing for RBI data
    rbi['Date'] = pd.to_datetime(rbi['Effective Date'], format='%d-%b-%y', errors='coerce')
    mask_na = rbi['Date'].isna()
    if mask_na.any():
        rbi.loc[mask_na, 'Date'] = pd.to_datetime(rbi.loc[mask_na, 'Effective Date'], format='%b-%y', errors='coerce')
    if rbi['Date'].isna().any():
        print("Dropping rows with unparseable dates in RBI data:")
        print(rbi[rbi['Date'].isna()])
        rbi = rbi.dropna(subset=['Date'])
    rbi = rbi.sort_values('Date').reset_index(drop=True)
    rbi = rbi.drop(columns=['Effective Date'])
    rbi['Repo Rate'] = rbi['Repo Rate'].astype(str).str.replace('%', '').str.strip().astype(float)

# Sentiment data
reddit = load_csv(os.path.join(BASE_DIR, 'data', 'reddit_nifty_sentiment.csv'))
twitter = load_csv(os.path.join(BASE_DIR, 'data', 'twitter_nifty_sentiment.csv'))
gdelt = load_csv(os.path.join(BASE_DIR, 'data', 'gdelt_nifty_news.csv'))

# --- Preprocess Each Data Source ---

# Clean and feature engineer market data
if nifty is not None:
    nifty = nifty.sort_values('Date').reset_index(drop=True)
    nifty = add_technical_indicators(nifty)
    nifty = add_lag_features(nifty, ['Close', 'Open', 'High', 'Low'])
    nifty = add_rolling_features(nifty, ['Close', 'Open', 'High', 'Low'])

if banknifty is not None:
    banknifty = banknifty.sort_values('Date').reset_index(drop=True)
    banknifty = add_technical_indicators(banknifty)
    banknifty = add_lag_features(banknifty, ['Close', 'Open', 'High', 'Low'])
    banknifty = add_rolling_features(banknifty, ['Close', 'Open', 'High', 'Low'])

# Preprocess sentiment data
reddit_daily = preprocess_sentiment(reddit, 'created_utc', 'sentiment') if reddit is not None else None
twitter_daily = preprocess_sentiment(twitter, 'created_at', 'sentiment') if twitter is not None else None

# Macro data: clean and align
if repo is not None:
    repo = repo.sort_values('Date').reset_index(drop=True)
    repo['Date'] = pd.to_datetime(repo['Date'], errors='coerce')
if rbi is not None:
    rbi = rbi.sort_values('Date').reset_index(drop=True)
    rbi['Date'] = pd.to_datetime(rbi['Date'], errors='coerce')

# --- Merge All Data on Date ---

main_df = nifty.copy() if nifty is not None else None

if main_df is not None:
    # Merge with macro data
    if repo is not None:
        main_df = merge_on_date(main_df, repo)
    if rbi is not None:
        main_df = merge_on_date(main_df, rbi)
    # Merge with sentiment
    if reddit_daily is not None:
        main_df = merge_on_date(main_df, reddit_daily)
    if twitter_daily is not None:
        main_df = merge_on_date(main_df, twitter_daily)
    # Merge with other data sources as available...

    # Sort and fill missing values
    main_df = main_df.sort_values('Date').reset_index(drop=True)
    main_df = main_df.ffill()
    main_df = main_df.fillna({col: 0 for col in main_df.select_dtypes(include='number').columns})

    # Optional: scale features
    # numeric_cols = main_df.select_dtypes(include=[np.number]).columns.tolist()
    # main_df = scale_features(main_df, numeric_cols, scaler_type='standard')

    # Save processed dataset
    os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
    main_df.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'nifty_merged_preprocessed.csv'), index=False)
    print("Preprocessing complete. Output saved to data/processed/nifty_merged_preprocessed.csv")
else:
    print("NIFTY data not found. Cannot proceed with preprocessing.")

banknifty_df = banknifty.copy() if banknifty is not None else None

if banknifty_df is not None:
    if repo is not None:
        banknifty_df = merge_on_date(banknifty_df, repo)
    if rbi is not None:
        banknifty_df = merge_on_date(banknifty_df, rbi)
    if reddit_daily is not None:
        banknifty_df = merge_on_date(banknifty_df, reddit_daily)
    if twitter_daily is not None:
        banknifty_df = merge_on_date(banknifty_df, twitter_daily)

    banknifty_df = banknifty_df.sort_values('Date').reset_index(drop=True)
    banknifty_df = banknifty_df.ffill()
    banknifty_df = banknifty_df.fillna({col: 0 for col in banknifty_df.select_dtypes(include='number').columns})

    os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
    banknifty_df.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'banknifty_merged_preprocessed.csv'), index=False)
    print("Preprocessing complete. Output saved to data/processed/banknifty_merged_preprocessed.csv")
else:
    print("BANKNIFTY data not found. Cannot proceed with preprocessing.")
