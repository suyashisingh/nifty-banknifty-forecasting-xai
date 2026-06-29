import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
INPUT_CSV  = os.path.join(MODELS_DIR, 'baseline_vs_naive.csv')
OUTPUT_PNG = os.path.join(MODELS_DIR, 'fig2_rmse_comparison.png')

MODELS      = ['Ensemble', 'XGBoost', 'Naive']
HORIZON_MAP = {1: 'Next-Day', 5: 'Next-Week'}
COLORS      = {'Ensemble': '#2196F3', 'XGBoost': '#FF9800', 'Naive': '#9E9E9E'}
INDICES     = ['NIFTY', 'BANKNIFTY']

df = pd.read_csv(INPUT_CSV)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
fig.suptitle('RMSE Comparison: Stacked Ensemble vs Baselines', fontsize=14, fontweight='bold', y=1.02)

x        = np.arange(len(HORIZON_MAP))
n_models = len(MODELS)
width    = 0.25
offsets  = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

handles = []
labels  = []

for ax, index in zip(axes, INDICES):
    sub = df[df['Index'] == index]
    for i, model in enumerate(MODELS):
        rmse_vals = []
        for h in sorted(HORIZON_MAP.keys()):
            row = sub[(sub['Model'] == model) & (sub['Horizon'] == h)]
            rmse_vals.append(float(row['RMSE'].values[0]))
        bars = ax.bar(x + offsets[i], rmse_vals, width, label=model,
                      color=COLORS[model], edgecolor='white', linewidth=0.6)
        if index == INDICES[0]:
            handles.append(bars)
            labels.append(model)

    ax.set_title(index, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_MAP[h] for h in sorted(HORIZON_MAP.keys())], fontsize=11)
    ax.set_xlabel('Horizon', fontsize=11)
    ax.set_ylabel('RMSE (index points)', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.0), frameon=False)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_PNG}")
