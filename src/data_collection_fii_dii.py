# import pandas as pd
# import os
#
# csv_path = 'data/raw/macro/nse_fii_dii_flows.csv'
#
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path, encoding='latin1')
#
#     # Clean column names (handle \xa0 and special chars)
#     df.columns = [
#         col.strip()
#         .replace('\xa0', '_')
#         .replace(' ', '_')
#         .replace('/', '_')
#         .replace('.', '_')
#         for col in df.columns
#     ]
#     print("Cleaned columns:", df.columns.tolist())
#
#     # Numeric columns based on your cleaned column names
#     num_cols = [
#         'Gross_Purchase',
#         'Gross_Sales',
#         'Net_Purchase___Sales',
#         'Gross_Purchase_1',
#         'Gross_Sales_1',
#         'Net_Purchase___Sales_1'
#     ]
#     for col in num_cols:
#         df[col] = (
#             df[col]
#             .astype(str)
#             .str.replace(',', '')
#             .str.replace('−', '-')
#             .astype(float)
#         )
#
#     # Robust date parsing for mixed formats
#     def parse_date(val):
#         for fmt in ('%d-%b-%y', '%d-%b-%Y', '%b-%y', '%b-%Y'):
#             try:
#                 return pd.to_datetime(val, format=fmt)
#             except Exception:
#                 continue
#         return pd.NaT
#
#     df['Date'] = df['Date'].apply(parse_date)
#
#     # Rename columns for clarity
#     df = df.rename(columns={
#         'Gross_Purchase': 'FII_Gross_Purchase',
#         'Gross_Sales': 'FII_Gross_Sales',
#         'Net_Purchase___Sales': 'FII_Net',
#         'Gross_Purchase_1': 'DII_Gross_Purchase',
#         'Gross_Sales_1': 'DII_Gross_Sales',
#         'Net_Purchase___Sales_1': 'DII_Net'
#     })
#
#     print("\nCleaned Data:")
#     print(df.head(10))
# else:
#     print(f"CSV not found at {csv_path}")


import pandas as pd
import os

# Define input and output paths
input_csv_path = 'data/raw/macro/nse_fii_dii_flows.csv'
output_dir = 'data/processed'
output_csv_path = os.path.join(output_dir, 'fii_dii_flows_cleaned.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(input_csv_path):
    df = pd.read_csv(input_csv_path, encoding='latin1')

    # Clean column names (handle \xa0 and special chars)
    df.columns = [
        col.strip()
        .replace('\xa0', '_')
        .replace(' ', '_')
        .replace('/', '_')
        .replace('.', '_')
        for col in df.columns
    ]
    print("Cleaned columns:", df.columns.tolist())

    # Define expected numeric columns after cleaning
    num_cols = [
        'Gross_Purchase',
        'Gross_Sales',
        'Net_Purchase___Sales',
        'Gross_Purchase_1',
        'Gross_Sales_1',
        'Net_Purchase___Sales_1'
    ]
    # Check for missing columns
    missing_cols = [col for col in num_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
    # Only process columns that exist
    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '')
                .str.replace('−', '-')  # Handle special minus sign
                .astype(float)
            )

    # Robust date parsing for mixed formats
    def parse_date(val):
        for fmt in ('%d-%b-%y', '%d-%b-%Y', '%b-%y', '%b-%Y'):
            try:
                return pd.to_datetime(val, format=fmt)
            except Exception:
                continue
        return pd.NaT

    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(parse_date)
        # Log any unparseable dates
        if df['Date'].isna().any():
            print("Warning: Some dates could not be parsed:")
            print(df[df['Date'].isna()])
    else:
        print("Error: 'Date' column not found in the input CSV.")

    # Rename columns for clarity if they exist
    rename_map = {
        'Gross_Purchase': 'FII_Gross_Purchase',
        'Gross_Sales': 'FII_Gross_Sales',
        'Net_Purchase___Sales': 'FII_Net',
        'Gross_Purchase_1': 'DII_Gross_Purchase',
        'Gross_Sales_1': 'DII_Gross_Sales',
        'Net_Purchase___Sales_1': 'DII_Net'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print("\nCleaned Data:")
    print(df.head(10))

    # Save the cleaned DataFrame
    df.to_csv(output_csv_path, index=False)
    print(f"\nCleaned data saved to: {output_csv_path}")

else:
    print(f"CSV not found at {input_csv_path}")
