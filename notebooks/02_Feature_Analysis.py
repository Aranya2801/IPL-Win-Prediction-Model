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
# 📈 IPL Win Prediction — Feature Analysis
## Notebook 02: Correlation, Distribution & Cricket Domain Insights

**Author:** Aranya2801
"""

# %% Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

plt.style.use("dark_background")
ROOT = Path("..").resolve()

# %% Load engineered data
df = pd.read_csv(ROOT / "data/processed/matches_engineered.csv")
print(f"Dataset shape: {df.shape}")
print(f"Win rate: {df['win'].mean():.3f}")
df.describe().T.round(3)

# %% Numerical Feature Distributions
NUM_COLS = [
    "runs_left", "balls_left", "wickets_left",
    "current_run_rate", "required_run_rate",
    "run_rate_diff", "pressure_index", "momentum_score",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    win0 = df[df["win"] == 0][col].dropna()
    win1 = df[df["win"] == 1][col].dropna()
    ax.hist(win0, bins=40, alpha=0.6, color="#e17055", label="Loss", density=True)
    ax.hist(win1, bins=40, alpha=0.6, color="#00b894", label="Win",  density=True)
    ax.set_title(col.replace("_", " ").title(), fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
plt.suptitle("Feature Distributions by Match Outcome", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(ROOT / "docs/images/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Correlation Heatmap
corr_cols = NUM_COLS + ["win"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT / "docs/images/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Win Rate by Over
win_by_over = df.groupby("over")["win"].mean().reset_index()
fig = px.line(win_by_over, x="over", y="win",
              title="Batting Team Win Rate by Over (2nd innings)",
              labels={"win": "Win Rate", "over": "Over"},
              template="plotly_dark", color_discrete_sequence=["#FFD700"])
fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4)
fig.add_vrect(x0=0, x1=6, fillcolor="#00B894", opacity=0.06, annotation_text="PP")
fig.add_vrect(x0=15, x1=20, fillcolor="#FF6B35", opacity=0.06, annotation_text="Death")
fig.update_layout(height=400, yaxis=dict(tickformat=".0%", range=[0.3, 0.7]))
fig.write_image(str(ROOT / "docs/images/win_rate_by_over.png"))
fig.show()

# %% Pressure Index vs Win Probability
pressure_bins = pd.cut(df["pressure_index"], bins=20)
pi_win = df.groupby(pressure_bins)["win"].mean().reset_index()
pi_win["pressure_mid"] = pi_win["pressure_index"].apply(lambda x: x.mid).astype(float)

fig = px.scatter(pi_win, x="pressure_mid", y="win",
                 title="Pressure Index vs Win Probability",
                 labels={"pressure_mid": "Pressure Index", "win": "Win Probability"},
                 template="plotly_dark", color="win",
                 color_continuous_scale="RdYlGn",
                 size=[10]*len(pi_win))
fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4)
fig.update_layout(height=400, yaxis=dict(tickformat=".0%"),
                  coloraxis_showscale=False)
fig.show()

# %% RRR vs Win Rate
rrr_bins = pd.cut(df["required_run_rate"].clip(0, 20), bins=20)
rrr_win = df.groupby(rrr_bins)["win"].agg(["mean", "count"]).reset_index()
rrr_win["rrr_mid"] = rrr_win["required_run_rate"].apply(lambda x: x.mid).astype(float)
rrr_win = rrr_win[rrr_win["count"] > 50]  # filter thin bins

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rrr_win["rrr_mid"], y=rrr_win["mean"],
    mode="lines+markers", line=dict(color="#FFD700", width=3),
    marker=dict(size=rrr_win["count"] / rrr_win["count"].max() * 20 + 4,
                color="#FF6B35"),
    name="Win Rate",
))
fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4)
fig.update_layout(
    title="Required Run Rate vs Win Probability (bubble size = sample count)",
    xaxis_title="Required Run Rate", yaxis_title="Win Probability",
    yaxis=dict(tickformat=".0%"),
    template="plotly_dark", height=420,
)
fig.show()

# %% Wickets Left vs Win Rate
wkt_win = df.groupby("wickets_left")["win"].mean().reset_index()
fig = px.bar(wkt_win, x="wickets_left", y="win",
             title="Win Rate by Wickets in Hand",
             labels={"wickets_left": "Wickets Left", "win": "Win Rate"},
             color="win", color_continuous_scale="RdYlGn",
             template="plotly_dark")
fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4)
fig.update_layout(height=400, yaxis=dict(tickformat=".0%"), coloraxis_showscale=False)
fig.write_image(str(ROOT / "docs/images/win_rate_wickets.png"))
fig.show()

# %% Feature vs Target: Point-Biserial Correlation
from scipy.stats import pointbiserialr
pb_corrs = {}
for col in NUM_COLS:
    vals = df[col].dropna()
    targets = df.loc[vals.index, "win"]
    r, p = pointbiserialr(targets, vals)
    pb_corrs[col] = {"r": r, "p": p, "significant": p < 0.05}

pb_df = pd.DataFrame(pb_corrs).T.sort_values("r", ascending=False)
print("\nPoint-Biserial Correlations with 'win':")
print(pb_df.to_string())

fig = px.bar(pb_df.reset_index(), x="index", y="r",
             color="r", color_continuous_scale="RdYlGn",
             title="Point-Biserial Correlation: Features vs Win",
             labels={"index": "Feature", "r": "Correlation (r)"},
             template="plotly_dark")
fig.update_layout(height=420, coloraxis_showscale=False,
                  xaxis_tickangle=-35)
fig.write_image(str(ROOT / "docs/images/feature_correlation.png"))
fig.show()
