import yfinance as yf
import pandas as pd
import os
from datetime import datetime
# Define ticker and date range
nifty_ticker = "^NSEI"
start_date = "2024-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download NIFTY data
print("Downloading NIFTY data from 2024-01-01 to today...")
nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date, interval='1d')
nifty_data.reset_index(inplace=True)

# Define your output directory and path
output_dir = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "nifty_merged_preprocessed2024.csv")

# Save to CSV
nifty_data.to_csv(output_path, index=False)
print(f"NIFTY data saved to {output_path}")

# Optional: Preview the first and last few rows
print(nifty_data.head())
print(nifty_data.tail())
