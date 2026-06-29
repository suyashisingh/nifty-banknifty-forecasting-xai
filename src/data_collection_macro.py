import pandas as pd
import os

os.makedirs('data/raw/macro', exist_ok=True)

# --- Load and clean RBI Repo Rate Data ---
repo_csv = 'data/raw/macro/rbi_repo_rate_history.csv'
if os.path.exists(repo_csv):
    repo_df = pd.read_csv(repo_csv)
    repo_df['Repo Rate'] = repo_df['Repo Rate'].str.rstrip('%').astype(float)
    repo_df['%Change'] = repo_df['%Change'].replace({'-': None}).str.rstrip('%')
    repo_df['%Change'] = pd.to_numeric(repo_df['%Change'], errors='coerce')

    # Handle mixed date formats
    repo_df['Effective Date'] = pd.to_datetime(
        repo_df['Effective Date'],
        format='%d-%b-%y',
        errors='coerce'
    ).fillna(
        pd.to_datetime(repo_df['Effective Date'], format='%b-%y', errors='coerce')
    )

    print("Cleaned RBI Repo Rate Data:")
    print(repo_df.head())
else:
    print(f"Repo rate CSV not found at {repo_csv}")

# --- Load and clean CPI Data (unchanged) ---
# ... [rest of your CPI code]
