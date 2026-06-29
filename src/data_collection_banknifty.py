# import yfinance as yf
# import pandas as pd
# import os
#
# # Download BANKNIFTY data
# banknifty_ticker = "^NSEBANK"
# start_date = "2010-01-01"
# end_date = None  # None will fetch up to today
#
# print("Downloading BANKNIFTY data...")
# banknifty_data = yf.download(banknifty_ticker, start=start_date, end=end_date, interval='1d')
# banknifty_data.reset_index(inplace=True)
# # Optional: Format date if needed
# # banknifty_data['Date'] = banknifty_data['Date'].dt.strftime('%d-%m-%Y')
#
# # Save to the expected path
# output_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\banknifty_merged_preprocessed1.csv"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# banknifty_data.to_csv(output_path, index=False)
# print(f"BANKNIFTY preprocessed data saved to {output_path}")


import yfinance as yf
import pandas as pd
import os
from datetime import datetime

banknifty_ticker = "^NSEBANK"
start_date = "2025-03-20"  # Only recent years!
end_date = datetime.today().strftime('%Y-%m-%d')

print("Downloading recent BANKNIFTY data...")
banknifty_data = yf.download(banknifty_ticker, start=start_date, end=end_date, interval='1d')
banknifty_data.reset_index(inplace=True)

output_path = r"D:\intern-25\NIFTY_BANKNIFTYPredictor\pythonProject\src\data\processed\banknifty_merged_preprocessed2025.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
banknifty_data.to_csv(output_path, index=False)
print(f"Recent BANKNIFTY data saved to {output_path}")
