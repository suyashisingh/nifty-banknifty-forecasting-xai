import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    from lime import lime_tabular
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# --- Paths ---
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
XAI_BASE_DIR = os.path.join(BASE_DIR, 'models', 'xai')

# --- Constants ---
TRAIN_RATIO       = 0.80
TOP_N_FEATURES    = 15
N_LIME_FEATURES   = 10
LIME_RANDOM_STATE = 42
DPI               = 150
FIG_SIZE          = (10, 6)

CONFIGURATIONS = [
    {
        'index':        'nifty',
        'horizon':      1,
        'data_file':    'nifty_engineered_features.csv',
        'model_prefix': 'nifty',
        'out_dir':      'xai_nifty_h1',
    },
    {
        'index':        'nifty',
        'horizon':      5,
        'data_file':    'nifty_engineered_features.csv',
        'model_prefix': 'nifty',
        'out_dir':      'xai_nifty_h5',
    },
    {
        'index':        'banknifty',
        'horizon':      1,
        'data_file':    'banknifty_engineered_features.csv',
        'model_prefix': 'banknifty',
        'out_dir':      'xai_banknifty_h1',
    },
    {
        'index':        'banknifty',
        'horizon':      5,
        'data_file':    'banknifty_engineered_features.csv',
        'model_prefix': 'banknifty',
        'out_dir':      'xai_banknifty_h5',
    },
]


class EnsembleWrapper:
    """Sklearn-compatible wrapper around the stacked ensemble predict function."""

    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, X):
        return self._predict_fn(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))


def run_configuration(cfg):
    idx    = cfg['index']
    h      = cfg['horizon']
    prefix = cfg['model_prefix']
    out_dir = os.path.join(XAI_BASE_DIR, cfg['out_dir'])
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Configuration: {idx.upper()} h={h}")
    print(f"{'=' * 60}")

    # --- STEP 1: Load artifacts ---
    data_path   = os.path.join(DATA_DIR,   cfg['data_file'])
    feat_path   = os.path.join(MODELS_DIR, f'{prefix}_feature_names.txt')
    scaler_path = os.path.join(MODELS_DIR, f'{prefix}_scaler_h{h}.pkl')
    xgb_path    = os.path.join(MODELS_DIR, f'{prefix}_xgb_h{h}.pkl')
    lgbm_path   = os.path.join(MODELS_DIR, f'{prefix}_lgbm_h{h}.pkl')
    cat_path    = os.path.join(MODELS_DIR, f'{prefix}_catboost_h{h}.pkl')
    ada_path    = os.path.join(MODELS_DIR, f'{prefix}_adaboost_h{h}.pkl')
    meta_path   = os.path.join(MODELS_DIR, f'{prefix}_meta_h{h}.pkl')

    for p in [data_path, feat_path, scaler_path, xgb_path, lgbm_path,
              cat_path, ada_path, meta_path]:
        if not os.path.exists(p):
            print(f"  ERROR: missing artifact: {p} -- skipping configuration")
            return

    with open(feat_path) as fh:
        feature_names = [line.strip() for line in fh if line.strip()]

    scaler     = joblib.load(scaler_path)
    xgb_model  = joblib.load(xgb_path)
    lgbm_model = joblib.load(lgbm_path)
    cat_model  = joblib.load(cat_path)
    ada_model  = joblib.load(ada_path)
    meta_model = joblib.load(meta_path)

    print(f"  Artifacts loaded: {len(feature_names)} features.")

    # --- STEP 2: Prepare data ---
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    X_raw = df[feature_names].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], 0)
    X_raw = X_raw.fillna(X_raw.mean())

    # Replicate target construction from training scripts
    target = (df['Close'].shift(-h) - df['Close']) / df['Close'] * 100
    X_raw  = X_raw.iloc[:-h].reset_index(drop=True)
    y_all  = target.iloc[:-h].values

    X_scaled  = scaler.transform(X_raw.values)
    split_idx = int(TRAIN_RATIO * len(X_scaled))

    X_train_s = X_scaled[:split_idx]
    X_test_s  = X_scaled[split_idx:]
    y_test    = y_all[split_idx:]
    instance  = X_test_s[-1]

    print(f"  Train: {len(X_train_s)} rows | Test: {len(X_test_s)} rows")

    # --- STEP 3: Ensemble predict function ---
    def ensemble_predict(X_array):
        p1 = xgb_model.predict(X_array)
        p2 = lgbm_model.predict(X_array)
        p3 = cat_model.predict(X_array)
        p4 = ada_model.predict(X_array)
        meta_input = np.column_stack([p1, p2, p3, p4])
        return meta_model.predict(meta_input)

    # --- STEP 4: SHAP (per base model) ---
    base_models = [
        ('xgb',      xgb_model,  'XGBoost'),
        ('lgbm',     lgbm_model, 'LightGBM'),
        ('catboost', cat_model,  'CatBoost'),
        ('adaboost', ada_model,  'AdaBoost'),
    ]

    shap_saved = 0
    for model_key, model, model_label in base_models:
        try:
            print(f"  SHAP: computing for {model_label} ...")
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_s)

            # Beeswarm (summary) plot
            shap.summary_plot(
                shap_values, X_test_s,
                feature_names=feature_names,
                max_display=TOP_N_FEATURES,
                show=False,
            )
            plt.gcf().set_size_inches(FIG_SIZE)
            beeswarm_path = os.path.join(
                out_dir, f'shap_{model_key}_{idx}_h{h}_beeswarm.png'
            )
            plt.savefig(beeswarm_path, dpi=DPI, bbox_inches='tight')
            plt.close('all')
            shap_saved += 1

            # Bar plot (mean absolute SHAP)
            shap.summary_plot(
                shap_values, X_test_s,
                feature_names=feature_names,
                plot_type='bar',
                max_display=TOP_N_FEATURES,
                show=False,
            )
            plt.gcf().set_size_inches(FIG_SIZE)
            bar_path = os.path.join(
                out_dir, f'shap_{model_key}_{idx}_h{h}_bar.png'
            )
            plt.savefig(bar_path, dpi=DPI, bbox_inches='tight')
            plt.close('all')
            shap_saved += 1

            print(f"    Saved beeswarm + bar for {model_label}.")

        except Exception as exc:
            print(f"    WARNING: SHAP failed for {model_label}: {exc} -- skipping.")

    # --- STEP 5: LIME (on full ensemble) ---
    print(f"  LIME: explaining last test instance ...")
    lime_path = os.path.join(out_dir, f'lime_{idx}_h{h}.png')
    try:
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_s,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=True,
            random_state=LIME_RANDOM_STATE,
        )
        lime_exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=ensemble_predict,
            num_features=N_LIME_FEATURES,
        )
        fig = lime_exp.as_pyplot_figure()
        fig.set_size_inches(FIG_SIZE)
        fig.savefig(lime_path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"    Saved: {lime_path}")
    except Exception as exc:
        print(f"    WARNING: LIME failed: {exc}")
        lime_path = 'failed'

    # --- STEP 6: ELI5 manual permutation importance (on full ensemble) ---
    print(f"  ELI5: computing permutation importance ...")
    eli5_path = os.path.join(out_dir, f'eli5_{idx}_h{h}.png')
    try:
        from sklearn.metrics import mean_squared_error

        baseline_preds = ensemble_predict(X_test_s)
        baseline_mse   = mean_squared_error(y_test, baseline_preds)

        importances = np.zeros(X_test_s.shape[1])
        for i in range(X_test_s.shape[1]):
            X_permuted      = X_test_s.copy()
            np.random.RandomState(42).shuffle(X_permuted[:, i])
            permuted_preds  = ensemble_predict(X_permuted)
            permuted_mse    = mean_squared_error(y_test, permuted_preds)
            importances[i]  = permuted_mse - baseline_mse

        # Top N features by importance, descending
        sorted_idx   = np.argsort(importances)[::-1][:TOP_N_FEATURES]
        top_features = [feature_names[i] for i in sorted_idx]
        top_imps     = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_imps[::-1], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel('Increase in MSE when feature is shuffled')
        ax.set_title(f'ELI5 Permutation Importance -- {idx.upper()} h={h}')
        plt.tight_layout()
        fig.savefig(eli5_path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"    Saved: {eli5_path}")
    except Exception as exc:
        print(f"    WARNING: ELI5 failed: {exc}")
        eli5_path = 'failed'

    # --- STEP 7: All outputs already written to out_dir above ---

    # --- STEP 8: Configuration summary ---
    print(f"\n  Configuration: {idx} h={h}")
    print(f"  SHAP: {shap_saved} plots saved")
    print(f"  LIME: saved to {lime_path}")
    print(f"  ELI5: saved to {eli5_path}")


if __name__ == '__main__':
    os.makedirs(XAI_BASE_DIR, exist_ok=True)

    for cfg in CONFIGURATIONS:
        run_configuration(cfg)

    print(f"\n{'=' * 60}")
    print("XAI COMPLETE")
    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print(f"All outputs saved to: {XAI_BASE_DIR}")
    print(f"{'=' * 60}")
