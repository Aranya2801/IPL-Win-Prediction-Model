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
# 📊 IPL Win Prediction — Exploratory Data Analysis
## Notebook 01: Data Profiling, Distribution Analysis & Cricket Insights

**Author:** Aranya2801  
**Version:** 2.0.0
"""

# %% Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (14, 6)
ROOT = Path("..").resolve()

# %% Load Data
matches   = pd.read_csv(ROOT / "data/raw/matches.csv")
deliveries = pd.read_csv(ROOT / "data/raw/deliveries.csv")

print(f"Matches   : {matches.shape}")
print(f"Deliveries: {deliveries.shape}")
matches.head(3)

# %% Basic Stats
print("=== MATCHES OVERVIEW ===")
print(matches.dtypes)
print("\nMissing values:\n", matches.isnull().sum()[matches.isnull().sum() > 0])

# %% Seasons
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
season_counts = matches.groupby("season").size()
season_counts.plot(kind="bar", ax=axes[0], color="#FFD700", edgecolor="none")
axes[0].set_title("Matches per Season", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Season"); axes[0].set_ylabel("Count")

# Toss decision
toss_dec = matches["toss_decision"].value_counts()
axes[1].pie(toss_dec.values, labels=toss_dec.index, autopct="%1.1f%%",
            colors=["#FF6B35","#00B894"], startangle=90)
axes[1].set_title("Toss Decision Split", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT / "docs/images/01_seasons_toss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Win Analysis
# Does winning toss help?
matches["toss_win_match_win"] = (matches["toss_winner"] == matches["winner"]).astype(int)
toss_advantage = matches["toss_win_match_win"].mean()
print(f"\nToss winner also won match: {toss_advantage:.1%}")

fig = px.bar(
    matches.groupby(["toss_decision","toss_win_match_win"]).size().reset_index(name="count"),
    x="toss_decision", y="count", color="toss_win_match_win",
    barmode="group", title="Toss Decision vs Match Win",
    color_discrete_map={0: "#e17055", 1: "#00b894"},
    template="plotly_dark"
)
fig.show()

# %% Team Performance
team_wins = matches["winner"].value_counts().head(12)
fig = px.bar(
    x=team_wins.values, y=team_wins.index,
    orientation="h", title="All-Time IPL Wins by Team",
    color=team_wins.values, color_continuous_scale="Viridis",
    template="plotly_dark"
)
fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Wins",
                  height=450, coloraxis_showscale=False)
fig.show()

# %% Score Distributions
# First innings totals
first_inn = deliveries[deliveries["inning"] == 1].groupby("match_id")["total_runs"].sum()
second_inn = deliveries[deliveries["inning"] == 2].groupby("match_id")["total_runs"].sum()

fig = go.Figure()
fig.add_trace(go.Histogram(x=first_inn, name="1st Innings",
                           marker_color="#FFD700", opacity=0.7, nbinsx=40))
fig.add_trace(go.Histogram(x=second_inn, name="2nd Innings",
                           marker_color="#FF6B35", opacity=0.7, nbinsx=40))
fig.update_layout(barmode="overlay", title="Score Distribution by Innings",
                  xaxis_title="Total Runs", yaxis_title="Frequency",
                  template="plotly_dark", height=400)
fig.add_vline(x=first_inn.mean(), line_dash="dash", line_color="#FFD700",
              annotation_text=f"1st avg: {first_inn.mean():.0f}")
fig.add_vline(x=second_inn.mean(), line_dash="dash", line_color="#FF6B35",
              annotation_text=f"2nd avg: {second_inn.mean():.0f}")
fig.show()

# %% Venue Analysis
venue_avg = (deliveries[deliveries["inning"] == 1]
             .merge(matches[["id","venue"]], left_on="match_id", right_on="id")
             .groupby("venue")["total_runs"].sum()
             / matches.groupby("venue").size()).dropna()

top_venues = venue_avg.nlargest(15).reset_index()
top_venues.columns = ["venue", "avg_score"]

fig = px.bar(top_venues, x="avg_score", y="venue", orientation="h",
             title="Average 1st Innings Score by Venue (Top 15)",
             color="avg_score", color_continuous_scale="RdYlGn",
             template="plotly_dark")
fig.update_layout(showlegend=False, yaxis_title="", height=500,
                  coloraxis_showscale=False)
fig.show()

# %% Wicket Patterns
wicket_deliveries = deliveries[deliveries["player_dismissed"].notna()]
wicket_by_over = wicket_deliveries.groupby("over").size() / matches.shape[0]

fig = px.area(x=wicket_by_over.index, y=wicket_by_over.values,
              title="Average Wickets per Over (Across All Matches)",
              labels={"x": "Over", "y": "Avg Wickets"},
              template="plotly_dark", color_discrete_sequence=["#FF6B35"])
fig.add_vrect(x0=0, x1=6, fillcolor="#00B894", opacity=0.08,
              annotation_text="Powerplay")
fig.add_vrect(x0=15, x1=20, fillcolor="#FF6B35", opacity=0.08,
              annotation_text="Death Overs")
fig.show()

# %% Run Rate Analysis
deliveries["balls_bowled"] = (deliveries["over"] * 6 + deliveries["ball"])
rr_by_over = (deliveries.groupby(["match_id","over"])["total_runs"].sum()
              .groupby("over").mean() / 1 * 6)  # per over

fig = px.line(x=rr_by_over.index, y=rr_by_over.values,
              title="Average Run Rate by Over",
              labels={"x": "Over", "y": "Run Rate (per over)"},
              template="plotly_dark")
fig.add_hrline(y=rr_by_over.mean(), line_dash="dash",
               annotation_text=f"Mean: {rr_by_over.mean():.1f}")
fig.show()

# %% Summary
print("\n" + "="*60)
print("  EDA SUMMARY")
print("="*60)
print(f"  Total matches        : {len(matches):,}")
print(f"  Total deliveries     : {len(deliveries):,}")
print(f"  Seasons covered      : {matches['season'].min()} – {matches['season'].max()}")
print(f"  Unique teams         : {matches['team1'].nunique()}")
print(f"  Unique venues        : {matches['venue'].nunique()}")
print(f"  Avg 1st innings score: {first_inn.mean():.1f}")
print(f"  Toss → win rate      : {toss_advantage:.1%}")
print("="*60)
