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
# 🧪 IPL Win Prediction — Model Experiments & Ablation Study
## Notebook 03: Comparing Models, Feature Groups, and Calibration

**Author:** Aranya2801
"""

# %% Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.calibration import CalibrationDisplay
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

plt.style.use("dark_background")
ROOT = Path("..").resolve()

# %% Load processed data
from src.models.train_model import FEATURE_COLS, TARGET_COL
from src.models.train_model import IPLEnsembleModel

df = pd.read_csv(ROOT / "data/processed/matches_engineered.csv")
model_obj = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
X = model_obj._prepare(df[FEATURE_COLS], fit=True)
y = df[TARGET_COL].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% Model Comparison
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    "XGBoost":             xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                             use_label_encoder=False, eval_metric="logloss",
                                             random_state=42, n_jobs=-1),
    "LightGBM":            lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                              random_state=42, verbose=-1, n_jobs=-1),
}

results = {}
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Log-Loss':>10}")
print("-" * 60)
for name, clf in models.items():
    acc  = cross_val_score(clf, X, y, cv=skf, scoring="accuracy",  n_jobs=-1).mean()
    auc  = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc",   n_jobs=-1).mean()
    ll   = -cross_val_score(clf, X, y, cv=skf, scoring="neg_log_loss", n_jobs=-1).mean()
    results[name] = {"accuracy": acc, "roc_auc": auc, "log_loss": ll}
    print(f"{name:<25} {acc:>10.4f} {auc:>10.4f} {ll:>10.4f}")

# %% Visualize Model Comparison
res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})

fig = go.Figure()
metrics = ["accuracy", "roc_auc"]
colors  = ["#FFD700", "#00B894"]
for metric, color in zip(metrics, colors):
    fig.add_trace(go.Bar(
        name=metric.replace("_", " ").title(),
        x=res_df["model"], y=res_df[metric],
        marker_color=color, opacity=0.85,
    ))
fig.update_layout(
    barmode="group",
    title="Model Comparison — Accuracy vs ROC-AUC (5-fold CV)",
    xaxis_title="Model", yaxis_title="Score",
    yaxis=dict(range=[0.6, 1.0]),
    template="plotly_dark", height=420,
)
fig.write_image(str(ROOT / "docs/images/model_comparison.png"))
fig.show()

# %% Ablation Study — Feature Groups
FEATURE_GROUPS = {
    "All Features": FEATURE_COLS,
    "No Run Rate":  [f for f in FEATURE_COLS if "run_rate" not in f],
    "No H2H":       [f for f in FEATURE_COLS if "h2h" not in f],
    "No Venue":     [f for f in FEATURE_COLS if "venue" not in f],
    "No Pressure":  [f for f in FEATURE_COLS if "pressure" not in f],
    "No Momentum":  [f for f in FEATURE_COLS if "momentum" not in f],
    "Only State":   ["runs_left", "balls_left", "wickets_left",
                     "current_run_rate", "required_run_rate"],
}

ablation = {}
best_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=42, n_jobs=-1)
print("\nAblation Study:")
print(f"{'Feature Group':<25} {'CV Accuracy':>12} {'Δ vs All':>10}")
print("-" * 50)
baseline = None
for group_name, feats in FEATURE_GROUPS.items():
    m = IPLEnsembleModel(feature_cols=feats, optimize=False)
    X_grp = m._prepare(df[feats], fit=True)
    acc = cross_val_score(best_clf, X_grp, y, cv=skf, scoring="accuracy", n_jobs=-1).mean()
    ablation[group_name] = acc
    if baseline is None: baseline = acc
    delta = acc - baseline
    print(f"{group_name:<25} {acc:>12.4f} {delta:>+10.4f}")

# %% Calibration Curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
uncal = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
calibrated = CalibratedClassifierCV(uncal, method="isotonic", cv=5)
uncal.fit(X_tr, y_tr)
calibrated.fit(X_tr, y_tr)

fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_estimator(uncal, X_te, y_te, n_bins=10, ax=ax, name="XGBoost (raw)")
CalibrationDisplay.from_estimator(calibrated, X_te, y_te, n_bins=10, ax=ax, name="XGBoost (calibrated)")
ax.set_title("Calibration Curve — Raw vs Isotonic Calibration", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(ROOT / "docs/images/calibration_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nCalibration Brier Scores:")
from sklearn.metrics import brier_score_loss
print(f"  Raw XGBoost  : {brier_score_loss(y_te, uncal.predict_proba(X_te)[:,1]):.4f}")
print(f"  Calibrated   : {brier_score_loss(y_te, calibrated.predict_proba(X_te)[:,1]):.4f}")
