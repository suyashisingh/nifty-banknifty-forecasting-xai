"""
Page 4 — XAI Explainability
Displays pre-computed SHAP/LIME/ELI5 PNG images for all four ensemble
configurations (NIFTY/BANKNIFTY × h=1/h=5) and the combined paper figures.
All images are loaded directly from models/xai/ — no on-the-fly computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import streamlit as st

BASE_DIR  = Path(__file__).parent.parent.parent   # pythonProject/
XAI_DIR   = BASE_DIR / "models" / "xai"
COMBINED_DIR = BASE_DIR / "models" / "xai_combined"

st.set_page_config(page_title="XAI Explainability", page_icon="🔍", layout="wide")
st.title("XAI Explainability")
st.caption(
    "All plots are pre-computed and saved as PNG files. "
    "Re-run `src/xai.py` to regenerate with updated model artifacts."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    index_label = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], key="xai_index")
    index = index_label.lower()
    horizon = st.selectbox("Horizon (days)", [1, 5], key="xai_horizon")

xai_folder = XAI_DIR / f"xai_{index}_h{horizon}"

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def show_image(path: Path, caption: str) -> None:
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: `{path.relative_to(BASE_DIR)}`")


# ---------------------------------------------------------------------------
# SHAP section
# ---------------------------------------------------------------------------

st.subheader(f"SHAP — {index_label} h={horizon}")

BASE_MODELS = ["xgb", "lgbm", "catboost"]
BASE_LABELS = {"xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost"}

shap_tabs = st.tabs([BASE_LABELS[m] for m in BASE_MODELS])

for tab, model_key in zip(shap_tabs, BASE_MODELS):
    with tab:
        label = BASE_LABELS[model_key]
        col_bee, col_bar = st.columns(2)

        beeswarm_path = xai_folder / f"shap_{model_key}_{index}_h{horizon}_beeswarm.png"
        bar_path      = xai_folder / f"shap_{model_key}_{index}_h{horizon}_bar.png"

        with col_bee:
            show_image(
                beeswarm_path,
                f"SHAP Beeswarm — {label}: shows the distribution of each feature's "
                "SHAP values across all test samples. Red = high feature value, "
                "blue = low. Horizontal spread shows impact magnitude and direction.",
            )
        with col_bar:
            show_image(
                bar_path,
                f"SHAP Bar — {label}: mean absolute SHAP value per feature. "
                "Ranks features by overall importance to the model's output.",
            )

st.caption(
    "AdaBoost SHAP is not shown — shap.TreeExplainer does not support "
    "AdaBoostRegressor and those plots were skipped during xai.py generation."
)

st.divider()

# ---------------------------------------------------------------------------
# LIME section
# ---------------------------------------------------------------------------

st.subheader(f"LIME — {index_label} h={horizon}")
lime_path = xai_folder / f"lime_{index}_h{horizon}.png"
show_image(
    lime_path,
    "LIME (Local Interpretable Model-agnostic Explanations): explains the "
    "full stacked ensemble's prediction on the last test instance. "
    "Each bar shows how much a discretized feature range pushed the prediction "
    "up (positive) or down (negative) from the local linear approximation's intercept.",
)

st.divider()

# ---------------------------------------------------------------------------
# ELI5 section
# ---------------------------------------------------------------------------

st.subheader(f"ELI5 Permutation Importance — {index_label} h={horizon}")
eli5_path = xai_folder / f"eli5_{index}_h{horizon}.png"
show_image(
    eli5_path,
    "ELI5 Permutation Importance: measures how much the ensemble's test-set MSE "
    "increases when each feature is randomly shuffled. Larger increase = more "
    "important feature. Computed over the full test split using the stacked ensemble.",
)

st.divider()

# ---------------------------------------------------------------------------
# Paper figures (combined grids)
# ---------------------------------------------------------------------------

with st.expander("Paper Figures (combined grids from xai_combined/)"):
    st.markdown("These are the multi-configuration panel figures used in the paper.")

    paper_figs = [
        ("fig7_shap_combined.png",         "Fig 7a — SHAP Bar (all 4 configurations)"),
        ("fig7_shap_beeswarm_combined.png", "Fig 7b — SHAP Beeswarm (all 4 configurations)"),
        ("fig8_lime_combined.png",          "Fig 8 — LIME (all 4 configurations)"),
        ("fig9_eli5_combined.png",          "Fig 9 — ELI5 Permutation Importance (all 4 configurations)"),
    ]

    for fname, caption in paper_figs:
        fpath = COMBINED_DIR / fname
        if fpath.exists():
            st.image(Image.open(fpath), caption=caption, use_container_width=True)
            st.divider()
        else:
            st.warning(f"Not found: `models/xai_combined/{fname}`")

# ---------------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------------

with st.expander(f"File inventory — models/xai/xai_{index}_h{horizon}/"):
    if xai_folder.exists():
        files = sorted(xai_folder.glob("*.png"))
        if files:
            inv = [{"File": f.name, "Size (KB)": f"{f.stat().st_size / 1024:.1f}"} for f in files]
            import pandas as pd
            st.dataframe(pd.DataFrame(inv), use_container_width=True, hide_index=True)
        else:
            st.info("No PNG files found in this directory.")
    else:
        st.error(f"Directory does not exist: {xai_folder}")
