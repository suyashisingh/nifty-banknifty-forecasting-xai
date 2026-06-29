import yfinance as yf
import pandas as pd
import os
os.makedirs('data/raw', exist_ok=True)



def fetch_data():
    # Fetch NIFTY 50 data (2010-2025 daily)
    nifty = yf.download(
        "^NSEI",
        start="2010-01-01",
        end="2025-01-01",
        interval="1d",
        repair=True  # Fixes common data issues [1][3]
    )

    # Fetch BANKNIFTY data
    banknifty = yf.download(
        "^NSEBANK",
        start="2010-01-01",
        end="2025-01-01",
        interval="1d",
        repair=True
    )

    # Save raw data
    nifty.to_csv("data/raw/nifty_raw.csv")
    banknifty.to_csv("data/raw/banknifty_raw.csv")

    return nifty, banknifty


if __name__ == "__main__":
    nifty_data, banknifty_data = fetch_data()
