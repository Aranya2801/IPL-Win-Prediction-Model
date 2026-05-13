# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# 🔍 IPL Win Prediction — SHAP Explainability Analysis
## Notebook 04: Feature Attribution, Decision Plots & Partial Dependence

**Author:** Aranya2801
"""

# %% Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

plt.style.use("dark_background")
ROOT = Path("..").resolve()

# %% Load model & data
from src.models.train_model import IPLEnsembleModel, FEATURE_COLS

model = IPLEnsembleModel.load(ROOT / "models/ipl_stacked_ensemble.pkl")

df = pd.read_csv(ROOT / "data/processed/matches_engineered.csv")
X_sample = df[FEATURE_COLS].head(500)
X_enc = model._prepare(X_sample, fit=False)

# %% SHAP TreeExplainer on XGBoost base learner
xgb_model = model.model.named_estimators_["xgb"]
explainer  = shap.TreeExplainer(xgb_model)
shap_vals  = explainer.shap_values(X_enc[:200])

# %% Summary Plot (Beeswarm)
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_vals, X_sample[:200],
    feature_names=FEATURE_COLS,
    plot_type="dot", show=False,
    max_display=15,
)
plt.title("SHAP Summary Plot — IPL Win Predictor", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT / "docs/images/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Bar Plot (Mean |SHAP|)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals, X_sample[:200],
    feature_names=FEATURE_COLS,
    plot_type="bar", show=False,
    max_display=15,
)
plt.title("Mean |SHAP| — Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT / "docs/images/shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Force Plot for a single prediction
shap.initjs()
idx = 42
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_vals[idx],
    X_sample.iloc[idx],
    feature_names=FEATURE_COLS,
)
shap.save_html(str(ROOT / "docs/images/shap_force_plot.html"), force_plot)
print(f"Force plot saved for prediction index {idx}")

# %% Dependence Plot — required_run_rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
shap.dependence_plot(
    "required_run_rate", shap_vals, X_sample[:200],
    feature_names=FEATURE_COLS, ax=axes[0], show=False,
    interaction_index="wickets_left",
)
axes[0].set_title("SHAP Dependence: RRR (colored by wickets left)")

shap.dependence_plot(
    "run_rate_diff", shap_vals, X_sample[:200],
    feature_names=FEATURE_COLS, ax=axes[1], show=False,
    interaction_index="pressure_index",
)
axes[1].set_title("SHAP Dependence: RR Diff (colored by pressure)")

plt.tight_layout()
plt.savefig(ROOT / "docs/images/shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Decision Plot
plt.figure(figsize=(10, 8))
shap.decision_plot(
    explainer.expected_value,
    shap_vals[:20],
    feature_names=FEATURE_COLS,
    show=False,
)
plt.title("SHAP Decision Plot (First 20 Predictions)", fontsize=14)
plt.tight_layout()
plt.savefig(ROOT / "docs/images/shap_decision.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Feature Ranking Table
importance = np.abs(shap_vals).mean(axis=0)
feat_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Mean_SHAP": importance,
    "Rank": range(1, len(FEATURE_COLS)+1),
}).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
feat_df.index += 1
print("\nTop 10 Features by Mean |SHAP|:")
print(feat_df.head(10).to_string())
