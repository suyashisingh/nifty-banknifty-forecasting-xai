import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_PNG = os.path.join(MODELS_DIR, 'fig5_prophet_forecast.png')

TRAIN_RATIO = 0.80
EVAL_DAYS   = 30

CONFIGURATIONS = [
    {
        'index':         'nifty',
        'data_file':     'nifty_merged_preprocessed.csv',
        'forecast_file': 'nifty_prophet_forecast.csv',
        'title':         'NIFTY',
        'train_ratio':   0.80,
    },
    {
        'index':         'banknifty',
        'data_file':     'banknifty_merged_preprocessed.csv',
        'forecast_file': 'banknifty_prophet_forecast.csv',
        'title':         'BANKNIFTY',
        'train_ratio':   0.90,
    },
]

COLORS = {
    'actual':     '#1565C0',
    'forecast':   '#E53935',
    'ci':         '#EF9A9A',
    'eval_band':  '#66BB6A',
}


def load_actual(cfg):
    path = os.path.join(BASE_DIR, 'data', 'processed', cfg['data_file'])
    raw  = pd.read_csv(path)
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw  = raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if 'Repo Rate' in raw.columns and 'Repo_Rate' not in raw.columns:
        raw = raw.rename(columns={'Repo Rate': 'Repo_Rate'})
    if 'sentiment_daily_x' in raw.columns and 'Reddit_Sentiment' not in raw.columns:
        raw = raw.rename(columns={'sentiment_daily_x': 'Reddit_Sentiment'})

    df = raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return df


def load_forecast(cfg):
    path = os.path.join(MODELS_DIR, cfg['forecast_file'])
    fc   = pd.read_csv(path)
    fc['ds'] = pd.to_datetime(fc['ds'])
    return fc


def eval_window_dates(df, train_ratio):
    split_idx  = int(train_ratio * len(df))
    eval_start = df.iloc[split_idx]['ds']
    eval_end   = df.iloc[min(split_idx + EVAL_DAYS - 1, len(df) - 1)]['ds']
    return eval_start, eval_end


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

handles_dict = {}

for ax, cfg in zip(axes, CONFIGURATIONS):
    # STEP 1 — actual data
    actual_df = load_actual(cfg)

    # STEP 2 — forecast
    forecast_df = load_forecast(cfg)

    # STEP 3 — eval window
    eval_start, eval_end = eval_window_dates(actual_df, cfg['train_ratio'])
    print(f"  {cfg['title']}: {len(actual_df)} actual rows | "
          f"eval window {eval_start.date()} to {eval_end.date()} | "
          f"forecast {forecast_df['ds'].min().date()} to {forecast_df['ds'].max().date()}")

    # STEP 4 — plot
    l_actual, = ax.plot(
        actual_df['ds'], actual_df['y'],
        color=COLORS['actual'], linewidth=1.1, linestyle='-', label='Actual',
    )
    l_forecast, = ax.plot(
        forecast_df['ds'], forecast_df['yhat'],
        color=COLORS['forecast'], linewidth=1.3, linestyle='--', label='Prophet Forecast',
    )
    ci = ax.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        color=COLORS['ci'], alpha=0.2, label='95% Confidence Interval',
    )
    eval_band = ax.axvspan(
        eval_start, eval_end,
        color=COLORS['eval_band'], alpha=0.15, label='30-Day Evaluation Window',
    )

    ax.set_title(cfg['title'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price (index points)', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(linestyle='--', alpha=0.30)
    ax.spines[['top', 'right']].set_visible(False)

    # Collect legend proxies from first subplot only (same across both)
    if not handles_dict:
        handles_dict = {
            'Actual':                    l_actual,
            'Prophet Forecast':          l_forecast,
            '95% Confidence Interval':   ci,
            '30-Day Evaluation Window':  eval_band,
        }

fig.suptitle('Prophet Model: Actual vs Forecast', fontsize=14,
             fontweight='bold', y=1.02)
fig.legend(
    list(handles_dict.values()),
    list(handles_dict.keys()),
    loc='upper center', ncol=4, fontsize=10,
    bbox_to_anchor=(0.5, 1.0), frameon=False,
)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PNG}")
