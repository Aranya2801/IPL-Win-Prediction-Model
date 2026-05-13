"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IPL WIN PREDICTION — Streamlit Frontend                            ║
║         Real-time Win Probability Dashboard with SHAP Explainability       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from streamlit_lottie import st_lottie
import altair as alt

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/Aranya2801/IPL-Win-Prediction-Model",
        "Report a bug":"https://github.com/Aranya2801/IPL-Win-Prediction-Model/issues",
        "About":       "IPL Win Prediction — MIT-Level ML Project by Aranya2801",
    }
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --ipl-blue:   #003f7f;
    --ipl-gold:   #ffd700;
    --ipl-orange: #ff6b35;
    --ipl-green:  #00b894;
    --ipl-red:    #e17055;
    --bg-dark:    #0a0e1a;
    --card-bg:    rgba(255,255,255,0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.big-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffd700 0%, #ff6b35 50%, #e056fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #8892a4;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,215,0,0.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(255,215,0,0.15);
}

.win-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    display: inline-block;
}

.batting-badge {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white;
}

.bowling-badge {
    background: linear-gradient(135deg, #e17055, #d63031);
    color: white;
}

.confidence-chip {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.conf-very-high { background:#00b89430; color:#00b894; border:1px solid #00b894; }
.conf-high      { background:#0984e330; color:#0984e3; border:1px solid #0984e3; }
.conf-medium    { background:#fdcb6e30; color:#fdcb6e; border:1px solid #fdcb6e; }
.conf-low       { background:#e1705530; color:#e17055; border:1px solid #e17055; }

.stButton > button {
    background: linear-gradient(135deg, #ffd700, #ff6b35) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 20px rgba(255,107,53,0.4) !important;
}

div[data-testid="stSlider"] .st-bv { background: #ffd700; }

.live-indicator {
    display: inline-flex; align-items: center; gap: 6px;
    color: #00b894; font-size: 0.8rem; font-weight: 600;
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #00b894;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.4); }
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders",
    "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
    "Royal Challengers Bangalore", "Sunrisers Hyderabad",
    "Gujarat Titans", "Lucknow Super Giants",
]

CITIES = [
    "Ahmedabad", "Bangalore", "Chennai", "Delhi", "Dharamsala",
    "Hyderabad", "Indore", "Jaipur", "Kolkata", "Lucknow",
    "Mumbai", "Mohali", "Pune",
]

TEAM_COLORS = {
    "Chennai Super Kings":           "#ffc72c",
    "Delhi Capitals":                "#004c97",
    "Kolkata Knight Riders":         "#3a225d",
    "Mumbai Indians":                "#004ba0",
    "Punjab Kings":                  "#ed1b24",
    "Rajasthan Royals":              "#ea1a85",
    "Royal Challengers Bangalore":   "#c8102e",
    "Sunrisers Hyderabad":           "#f7a721",
    "Gujarat Titans":                "#1c2951",
    "Lucknow Super Giants":          "#a72b2a",
}

TEAM_LOGOS = {
    "Chennai Super Kings":           "🦁",
    "Delhi Capitals":                "🦅",
    "Kolkata Knight Riders":         "⚔️",
    "Mumbai Indians":                "🔵",
    "Punjab Kings":                  "👑",
    "Rajasthan Royals":              "💎",
    "Royal Challengers Bangalore":   "🔴",
    "Sunrisers Hyderabad":           "🌅",
    "Gujarat Titans":                "🏛️",
    "Lucknow Super Giants":          "🌆",
}


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def call_api(payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def local_predict(state: dict) -> dict:
    """Fallback local prediction when API is offline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from src.models.train_model import IPLEnsembleModel
        import pandas as pd

        model = IPLEnsembleModel.load()
        balls_done = int(state["overs_completed"]) * 6 + round((state["overs_completed"] % 1) * 10)
        balls_left = max(0, 120 - balls_done)
        runs_left  = max(0, state["target"] - state["current_score"])
        crr = state["current_score"] / (balls_done / 6) if balls_done else 0
        rrr = runs_left / (balls_left / 6) if balls_left else 99

        features = pd.DataFrame([{
            "batting_team":                 state["batting_team"],
            "bowling_team":                 state["bowling_team"],
            "city":                         state["city"],
            "runs_left":                    runs_left,
            "balls_left":                   balls_left,
            "wickets_left":                 10 - state["wickets_fallen"],
            "target":                       state["target"],
            "current_run_rate":             crr,
            "required_run_rate":            rrr,
            "total_runs_x":                 state["current_score"],
            "avg_powerplay_runs":           42.0,
            "batting_team_win_rate":        state.get("batting_team_win_rate", 0.5),
            "bowling_team_win_rate":        state.get("bowling_team_win_rate", 0.5),
            "batting_team_avg_score":       165.0,
            "bowling_team_avg_score":       163.0,
            "venue_avg_score":              168.0,
            "venue_win_batting_first_rate": state.get("venue_win_batting_first_rate", 0.5),
            "h2h_batting_win_rate":         state.get("h2h_batting_win_rate", 0.5),
            "last5_batting_win_rate":       state.get("batting_team_win_rate", 0.5),
            "last5_bowling_win_rate":       state.get("bowling_team_win_rate", 0.5),
            "wicket_phase":                 state["wickets_fallen"],
            "run_rate_diff":                crr - rrr,
            "pressure_index":               (runs_left / (balls_left + 1e-6)) * (1 + state["wickets_fallen"] / 10),
            "momentum_score":               (crr - rrr) * 0.5,
            "innings":                      2,
        }])
        proba = model.predict_proba(features)[0]
        bp = float(proba[1])
        return {
            "batting_team":      state["batting_team"],
            "bowling_team":      state["bowling_team"],
            "batting_win_prob":  round(bp, 4),
            "bowling_win_prob":  round(1 - bp, 4),
            "predicted_winner":  state["batting_team"] if bp >= 0.5 else state["bowling_team"],
            "confidence":        "High" if abs(bp - 0.5) > 0.2 else "Medium",
            "runs_left":         runs_left,
            "balls_left":        balls_left,
            "wickets_left":      10 - state["wickets_fallen"],
            "current_run_rate":  round(crr, 2),
            "required_run_rate": round(rrr, 2),
            "run_rate_diff":     round(crr - rrr, 2),
            "pressure_index":    round((runs_left / (balls_left + 1e-6)) * (1 + state["wickets_fallen"] / 10), 2),
            "top_features":      {},
            "model_version":     "2.0.0",
            "inference_time_ms": 0,
            "cache_hit":         False,
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def build_gauge(prob: float, team: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": f"{TEAM_LOGOS.get(team, '🏏')} {team}", "font": {"size": 14, "family": "Rajdhani"}},
        delta={"reference": 50, "suffix": "%"},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar":   {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(255,107,53,0.15)"},
                {"range": [30, 50], "color": "rgba(253,203,110,0.15)"},
                {"range": [50, 70], "color": "rgba(0,184,148,0.15)"},
                {"range": [70,100], "color": "rgba(0,184,148,0.25)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
        number={"suffix": "%", "font": {"size": 36, "family": "Rajdhani", "color": color}},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def build_win_prob_timeline(history: List[dict]) -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    bt = df["batting_team"].iloc[0] if "batting_team" in df else "Batting Team"
    bwt = df["bowling_team"].iloc[0] if "bowling_team" in df else "Bowling Team"
    bc = TEAM_COLORS.get(bt, "#00b894")
    bwc = TEAM_COLORS.get(bwt, "#e17055")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["over"], y=df["batting_prob"] * 100,
        mode="lines+markers", name=bt,
        line=dict(color=bc, width=3),
        marker=dict(size=6),
        fill="tozeroy", fillcolor=f"rgba{tuple(int(bc.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.15,)}",
    ))
    fig.add_trace(go.Scatter(
        x=df["over"], y=df["bowling_prob"] * 100,
        mode="lines+markers", name=bwt,
        line=dict(color=bwc, width=3, dash="dot"),
        marker=dict(size=6),
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="50%", annotation_position="left")
    fig.update_layout(
        title="Win Probability Timeline",
        xaxis_title="Over",
        yaxis_title="Win Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis_gridcolor="rgba(255,255,255,0.05)",
    )
    return fig


def build_pressure_chart(runs_left: int, balls_left: int,
                         wickets_left: int, rrr: float) -> go.Figure:
    overs_left = balls_left / 6
    x = np.linspace(0, overs_left, 50)
    y_needed = [runs_left - (rrr * xi * 6 / 6) for xi in x]
    y_needed = [max(0, v) for v in y_needed]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_needed, mode="lines",
        name="Runs Required", fill="tozeroy",
        line=dict(color="#ffd700", width=2),
        fillcolor="rgba(255,215,0,0.1)",
    ))
    fig.update_layout(
        title=f"Runs Required Trajectory | {wickets_left} wickets left",
        xaxis_title="Overs Remaining",
        yaxis_title="Runs Required",
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis_gridcolor="rgba(255,255,255,0.05)",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # ── header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="big-title">🏏 IPL WIN PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time Win Probability · Stacked Ensemble ML · SHAP Explainability</div>',
                unsafe_allow_html=True)

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Match Setup")
        st.markdown('<div class="live-indicator"><div class="live-dot"></div>LIVE MODE</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        batting_team = st.selectbox("🏏 Batting Team", TEAMS, index=3)
        bowling_team = st.selectbox("🎳 Bowling Team",
                                    [t for t in TEAMS if t != batting_team], index=0)
        city = st.selectbox("📍 Venue City", CITIES, index=9)

        st.markdown("---")
        st.markdown("### 🎯 Match State")
        target = st.number_input("Target Score", min_value=50, max_value=350, value=185)
        current_score = st.number_input("Current Score", min_value=0, max_value=350, value=87)
        wickets_fallen = st.slider("Wickets Fallen", 0, 10, 3)

        overs_int = st.slider("Overs Completed", 0, 19, 10)
        balls_extra = st.slider("+ Balls", 0, 5, 3)
        overs_completed = overs_int + balls_extra / 10

        st.markdown("---")
        st.markdown("### 📊 Historical Stats (Optional)")
        batting_wr = st.slider("Batting Team Win Rate", 0.0, 1.0, 0.55, 0.01)
        bowling_wr = st.slider("Bowling Team Win Rate", 0.0, 1.0, 0.50, 0.01)
        h2h_wr     = st.slider("H2H Batting Win Rate", 0.0, 1.0, 0.52, 0.01)
        venue_rate = st.slider("Venue: Batting 1st Win Rate", 0.0, 1.0, 0.48, 0.01)

        st.markdown("---")
        predict_btn = st.button("⚡ PREDICT NOW", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Simulation Mode")
        sim_mode = st.toggle("Enable Over-by-Over Simulation")

    # ── prediction ────────────────────────────────────────────────────────────
    if predict_btn or "result" not in st.session_state:
        state = {
            "batting_team":                 batting_team,
            "bowling_team":                 bowling_team,
            "city":                         city,
            "target":                       target,
            "current_score":                current_score,
            "wickets_fallen":               wickets_fallen,
            "overs_completed":              overs_completed,
            "batting_team_win_rate":        batting_wr,
            "bowling_team_win_rate":        bowling_wr,
            "h2h_batting_win_rate":         h2h_wr,
            "venue_win_batting_first_rate": venue_rate,
        }
        with st.spinner("🤖 Running ensemble inference…"):
            result = call_api(state) or local_predict(state)
        st.session_state["result"] = result
        st.session_state["state"]  = state

        # History for timeline
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "over":          overs_completed,
            "batting_prob":  result["batting_win_prob"] if result else 0.5,
            "bowling_prob":  result["bowling_win_prob"] if result else 0.5,
            "batting_team":  batting_team,
            "bowling_team":  bowling_team,
        })

    result = st.session_state.get("result")
    if not result:
        st.warning("⚠️ Could not get prediction. Ensure API is running or model is trained.")
        return

    # ── main grid ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    bc = TEAM_COLORS.get(result["batting_team"], "#00b894")
    bwc = TEAM_COLORS.get(result["bowling_team"], "#e17055")

    with col1:
        st.plotly_chart(build_gauge(result["batting_win_prob"],
                                    result["batting_team"], bc),
                        use_container_width=True)
    with col2:
        st.plotly_chart(build_gauge(result["bowling_win_prob"],
                                    result["bowling_team"], bwc),
                        use_container_width=True)

    # Winner banner
    winner = result["predicted_winner"]
    conf   = result["confidence"]
    conf_class = f"conf-{conf.lower().replace(' ', '-')}"
    winner_color = bc if winner == result["batting_team"] else bwc
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:0.9rem;color:#8892a4;
                    text-transform:uppercase;letter-spacing:2px;margin-bottom:0.3rem;">
            Predicted Winner
        </div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
                    color:{winner_color};">
            {TEAM_LOGOS.get(winner,'🏏')} {winner}
        </div>
        <span class="confidence-chip {conf_class}">{conf} Confidence</span>
        &nbsp;&nbsp;
        <span style="color:#8892a4;font-size:0.8rem;">
            {result.get('inference_time_ms',0):.1f}ms
            {'  ⚡ Cached' if result.get('cache_hit') else ''}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Runs Required",  result["runs_left"])
    with m2:
        st.metric("Balls Left",     result["balls_left"])
    with m3:
        st.metric("Wickets Left",   result["wickets_left"])
    with m4:
        crr, rrr = result["current_run_rate"], result["required_run_rate"]
        st.metric("CRR / RRR",      f"{crr:.2f} / {rrr:.2f}",
                  delta=f"{crr-rrr:+.2f}")
    with m5:
        st.metric("Pressure Index", f"{result['pressure_index']:.2f}")

    st.markdown("---")

    # ── charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        history = st.session_state.get("history", [])
        st.plotly_chart(build_win_prob_timeline(history), use_container_width=True)

    with col_b:
        st.plotly_chart(
            build_pressure_chart(
                result["runs_left"], result["balls_left"],
                result["wickets_left"], result["required_run_rate"]
            ),
            use_container_width=True
        )

        # SHAP top features
        if result.get("top_features"):
            feat_df = pd.DataFrame(
                list(result["top_features"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_feat = go.Figure(go.Bar(
                x=feat_df["Importance"], y=feat_df["Feature"],
                orientation="h",
                marker=dict(
                    color=feat_df["Importance"],
                    colorscale=[[0, "#e17055"], [0.5, "#fdcb6e"], [1, "#00b894"]],
                )
            ))
            fig_feat.update_layout(
                title="🔍 SHAP Feature Importance",
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", size=11),
                yaxis_gridcolor="rgba(0,0,0,0)",
                xaxis_gridcolor="rgba(255,255,255,0.05)",
                margin=dict(l=0, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    # ── reset timeline ────────────────────────────────────────────────────────
    if st.button("🔄 Reset Timeline", use_container_width=False):
        st.session_state["history"] = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.75rem;'>"
        "IPL Win Predictor v2.0 · Stacked Ensemble (XGBoost + LightGBM + RF + GB) · "
        "Bayesian Optimized · SHAP Explainability · "
        "<a href='https://github.com/Aranya2801/IPL-Win-Prediction-Model' "
        "style='color:#ffd700;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
