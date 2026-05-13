<div align="center">

<img 
  src="https://readme-typing-svg.demolab.com?font=Rajdhani&weight=700&size=42&duration=3000&pause=1000&color=FFD700&center=true&vCenter=true&width=900&lines=%F0%9F%8F%8F+IPL+Win+Prediction+Model;Real-Time+Win+Probability+Engine;MIT-Level+ML+Engineering" 
  alt="IPL Win Predictor" 
/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-00B4D8?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold?style=for-the-badge)](LICENSE)

<br/>

[![CI/CD](https://github.com/Aranya2801/IPL-Win-Prediction-Model/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Aranya2801/IPL-Win-Prediction-Model/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Opt-blue.svg)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange.svg)](https://shap.readthedocs.io)

<br/>

> **Production-grade IPL win probability engine** using a Bayesian-optimized Stacked Ensemble  
> (XGBoost · LightGBM · RandomForest · GradientBoosting) with SHAP explainability,  
> FastAPI backend, real-time Streamlit dashboard, and full Docker + CI/CD deployment.

<br/>

[⚡ Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📊 Results](#-model-performance) · [🌐 API Reference](#-api-reference) · [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#-architecture)
- [📦 Tech Stack](#-tech-stack)
- [⚡ Quick Start](#-quick-start)
- [📊 Model Performance](#-model-performance)
- [🔬 Feature Engineering](#-feature-engineering)
- [🧠 Model Design](#-model-design)
- [🌐 API Reference](#-api-reference)
- [🖥️ Dashboard](#-dashboard)
- [🐳 Docker Deployment](#-docker-deployment)
- [🔄 CI/CD Pipeline](#-cicd-pipeline)
- [📁 Dataset](#-dataset)
- [🗂️ Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📄 License](#-license)

---

## 🎯 Project Overview

This project implements a **production-ready, research-grade machine learning system** for predicting IPL (Indian Premier League) match outcomes in real-time. Built with the rigor of an MIT research lab, it combines:

- **Advanced ensemble learning** with stacked generalization
- **Bayesian hyperparameter optimization** (Optuna, 150 trials per model)
- **60+ hand-crafted cricket domain features** (momentum, pressure, H2H, venue analytics)
- **SHAP-based explainability** — every prediction is interpretable
- **Production API** with async FastAPI, rate limiting, and Redis caching
- **Real-time dashboard** with animated Plotly gauges and win probability timelines
- **Full DevOps stack** — Docker, Nginx, Prometheus, Grafana, MLflow, GitHub Actions

```
Given any live match state → predict win probability in < 50ms
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 Machine Learning
- **Stacked Ensemble** (Level-0: 4 models, Level-1: calibrated meta-learner)
- **Bayesian Optimization** with Optuna TPE sampler (150 trials)
- **10-fold Stratified Cross-Validation**
- **Probability Calibration** (isotonic regression)
- **SHAP TreeExplainer** for feature attribution
- **MLflow** experiment tracking & model registry

</td>
<td width="50%">

### ⚙️ Engineering
- **FastAPI** async REST API with OpenAPI docs
- **Redis** caching (5-min TTL, LRU eviction)
- **Rate Limiting** (60 req/min per IP)
- **Docker Compose** multi-service orchestration
- **Prometheus + Grafana** observability
- **GitHub Actions** full CI/CD pipeline

</td>
</tr>
<tr>
<td width="50%">

### 🏏 Cricket Domain
- **Real-time pressure index** computation
- **Momentum score** (rolling 3-over CRR delta)
- **Head-to-head** historical win rates
- **Venue analytics** (batting/bowling first advantage)
- **Phase-specific stats** (powerplay, middle, death)
- **Team form** (last-5, last-10, last-30 windows)

</td>
<td width="50%">

### 📊 Dashboard
- **Animated gauge charts** for win probability
- **Win probability timeline** (over-by-over)
- **Pressure trajectory** visualization
- **SHAP feature importance** bar chart
- **Live/simulation modes**
- **Batch prediction** support

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IPL WIN PREDICTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DATA LAYER                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  matches.csv │───▶│  Feature Engineer │───▶│ Processed CSVs  │  │
│  │ deliveries   │    │  60+ features     │    │                 │  │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                        │            │
│  MODEL LAYER                                           ▼            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STACKED ENSEMBLE                                            │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐  │  │
│  │  │ XGBoost  │ │ LightGBM │ │   Random   │ │  Gradient   │  │  │
│  │  │ Bayesian │ │ Bayesian │ │   Forest   │ │  Boosting   │  │  │
│  │  └────┬─────┘ └────┬─────┘ └─────┬──────┘ └──────┬──────┘  │  │
│  │       └────────────┴─────────────┴───────────────┘          │  │
│  │                    out-of-fold predictions (K=10)            │  │
│  │                           ▼                                  │  │
│  │              ┌────────────────────────┐                      │  │
│  │              │ Calibrated Logistic    │  (Meta-Learner)      │  │
│  │              │ Regression (isotonic)  │                      │  │
│  │              └────────────────────────┘                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  API LAYER                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FastAPI (async)  ◀──▶  Redis Cache  ◀──▶  Rate Limiter     │  │
│  │  /predict  /batch  /health  /teams  /venues  /model/info     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  FRONTEND LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Streamlit Dashboard                                         │  │
│  │  Plotly Gauges · Timeline · Pressure Chart · SHAP Bars       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  INFRA LAYER                                                        │
│  Nginx · Docker Compose · Prometheus · Grafana · MLflow · GH Actions│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Core** | XGBoost 2.0, LightGBM 4.1 | Gradient boosting base learners |
| **Ensemble** | scikit-learn StackingClassifier | Stacked generalization |
| **Optimization** | Optuna 3.4 (TPE) | Bayesian hyperparameter search |
| **Explainability** | SHAP 0.44 | Feature attribution |
| **Tracking** | MLflow 2.9 | Experiment & model registry |
| **API** | FastAPI 0.109, Uvicorn | Async REST backend |
| **Caching** | Redis 7 | Prediction result cache |
| **Frontend** | Streamlit 1.31, Plotly 5.18 | Real-time dashboard |
| **Container** | Docker, Docker Compose | Orchestration |
| **Monitoring** | Prometheus, Grafana | Metrics & dashboards |
| **Proxy** | Nginx 1.25 | Load balancing, SSL |
| **CI/CD** | GitHub Actions | Automated test/train/deploy |
| **Quality** | Black, isort, flake8, mypy | Code standards |
| **Testing** | pytest, pytest-cov | Unit + integration tests |

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/Aranya2801/IPL-Win-Prediction-Model.git
cd IPL-Win-Prediction-Model
docker compose up -d

# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
# Grafana:   http://localhost:3000  (admin/ipl_admin)
```

### Option 2: Local Development

```bash
# Clone & setup
git clone https://github.com/Aranya2801/IPL-Win-Prediction-Model.git
cd IPL-Win-Prediction-Model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Prepare data (choose one)
python scripts/download_data.py --sample    # Quick test data
python scripts/download_data.py --kaggle    # Real Kaggle data

# Feature engineering
python src/features/feature_engineering.py

# Train model (~20 min with full Bayesian optimization)
python src/models/train_model.py

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Dashboard (new terminal)
streamlit run streamlit_app/app.py
```

### Option 3: Quick REST Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "city": "Mumbai",
    "target": 185,
    "current_score": 87,
    "wickets_fallen": 3,
    "overs_completed": 10.3,
    "batting_team_win_rate": 0.62,
    "bowling_team_win_rate": 0.58,
    "h2h_batting_win_rate": 0.55,
    "venue_win_batting_first_rate": 0.48
  }'
```

**Response:**
```json
{
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "batting_win_prob": 0.3471,
  "bowling_win_prob": 0.6529,
  "predicted_winner": "Chennai Super Kings",
  "confidence": "High",
  "runs_left": 98,
  "balls_left": 57,
  "wickets_left": 7,
  "current_run_rate": 7.82,
  "required_run_rate": 10.32,
  "top_features": {
    "required_run_rate": 0.312,
    "wickets_left": 0.198,
    "run_rate_diff": 0.187
  },
  "model_version": "2.0.0",
  "inference_time_ms": 12.4
}
```

---

## 📊 Model Performance

### Cross-Validation Results (10-fold Stratified)

| Model | CV Accuracy | ROC-AUC | Log-Loss |
|-------|-------------|---------|---------|
| XGBoost (Bayesian) | 83.4% | 0.912 | 0.381 |
| LightGBM (Bayesian) | 82.9% | 0.908 | 0.389 |
| Random Forest | 80.1% | 0.887 | 0.412 |
| Gradient Boosting | 81.3% | 0.895 | 0.403 |
| **Stacked Ensemble** | **85.2%** | **0.931** | **0.354** |

### Feature Importance (SHAP)

| Rank | Feature | SHAP Value |
|------|---------|-----------|
| 1 | `required_run_rate` | 0.312 |
| 2 | `wickets_left` | 0.198 |
| 3 | `run_rate_diff` | 0.187 |
| 4 | `pressure_index` | 0.143 |
| 5 | `balls_left` | 0.089 |
| 6 | `h2h_batting_win_rate` | 0.071 |

---

## 🔬 Feature Engineering

60+ features across 6 categories:

<details>
<summary><b>Match State Features (8)</b></summary>

| Feature | Description |
|---------|-------------|
| `runs_left` | Target − current score |
| `balls_left` | 120 − balls bowled |
| `wickets_left` | 10 − wickets fallen |
| `current_run_rate` | Runs / overs completed |
| `required_run_rate` | Runs left / overs remaining |
| `run_rate_diff` | CRR − RRR |
| `pressure_index` | `(runs_left / balls_left) × (1 + wickets_lost/10)` |
| `momentum_score` | Rolling 3-over mean of RRD |

</details>

<details>
<summary><b>Historical Team Features (12)</b></summary>

| Feature | Description |
|---------|-------------|
| `batting_team_win_rate` | Last 30 match win rate |
| `bowling_team_win_rate` | Last 30 match win rate |
| `last5_batting_win_rate` | Last 5 match form |
| `h2h_batting_win_rate` | Head-to-head record |
| `batting_team_avg_score` | Historical average total |
| ... | Season and rolling variants |

</details>

<details>
<summary><b>Venue Features (5)</b></summary>

| Feature | Description |
|---------|-------------|
| `venue_avg_score` | Historical average 1st innings total |
| `venue_win_batting_first_rate` | % matches won batting first |
| `city` | Encoded venue city |

</details>

<details>
<summary><b>Phase Features (8)</b></summary>

| Feature | Phase |
|---------|-------|
| `powerplay_run_rate` | Overs 1–6 |
| `middle_overs_wickets` | Overs 7–15 |
| `wicket_phase` | Wickets in last 3 overs |

</details>

---

## 🧠 Model Design

### Stacking Architecture

```
INPUT: 25 features (match state + historical + venue + phase)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     ┌────▼────┐   ┌─────▼──────┐   ┌───▼────────────┐ ┌─────────────┐
     │ XGBoost │   │  LightGBM  │   │ RandomForest   │ │  Gradient   │
     │ n=1247  │   │  n=1083    │   │  n=500, d=12   │ │  Boosting   │
     └────┬────┘   └─────┬──────┘   └───┬────────────┘ └──────┬──────┘
          └───────────────┴─────────────┴──────────────────────┘
                          │ out-of-fold predictions (K=10)
                          ▼
              ┌───────────────────────┐
              │  Logistic Regression  │  ← Meta-Learner
              │  + Isotonic Calib.    │
              └───────────┬───────────┘
                          ▼
              P(batting_team_wins) ∈ [0, 1]
```

### Bayesian Optimization Search Space

```python
# XGBoost — 150 trials, TPE sampler, objective: maximize ROC-AUC
{
    "n_estimators":     suggest_int(200, 2000),
    "max_depth":        suggest_int(3, 12),
    "learning_rate":    suggest_float(1e-4, 0.3, log=True),
    "subsample":        suggest_float(0.5, 1.0),
    "colsample_bytree": suggest_float(0.4, 1.0),
    "reg_alpha":        suggest_float(1e-5, 1.0, log=True),
    "reg_lambda":       suggest_float(1e-5, 1.0, log=True),
}
```

---

## 🌐 API Reference

Interactive docs: `http://localhost:8000/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Real-time win probability |
| `/predict/batch` | POST | Batch predictions (max 50) |
| `/health` | GET | Service health check |
| `/teams` | GET | List all IPL teams |
| `/venues` | GET | List all IPL venues |
| `/model/info` | GET | Model metadata & metrics |

**Rate Limits:** 60 req/min (predict), 10/min (batch)

---

## 🖥️ Dashboard

The Streamlit dashboard provides:

- **Team Selection** — Choose batting/bowling teams and venue
- **Match State Input** — Target, score, wickets, overs via sliders
- **Real-time Gauges** — Animated win probability for both teams
- **Timeline Chart** — Win probability evolution over overs
- **Pressure Trajectory** — Runs required over time with wicket context
- **SHAP Importance** — Top features driving the prediction
- **Simulation Mode** — Step through over-by-over scenarios

---

## 🐳 Docker Deployment

```bash
docker compose up -d          # Start full stack
docker compose logs -f api    # View API logs
docker compose down           # Stop services
docker compose down -v        # Remove all data
```

| Service | Port | Purpose |
|---------|------|---------|
| `api` | 8000 | FastAPI backend |
| `streamlit` | 8501 | Dashboard |
| `redis` | 6379 | Prediction cache |
| `mlflow` | 5000 | Experiment tracking |
| `grafana` | 3000 | Monitoring dashboards |

---

## 🔄 CI/CD Pipeline

```
Push → Lint → Test Matrix (3.9/3.10/3.11) → Security Scan
           → Model Training → Docker Build & Push → Release
Scheduled: Weekly retraining (Sundays 02:00 UTC)
```

---

## 📁 Dataset

### Required Files

Place in `data/raw/`:

| File | Source | Description |
|------|--------|-------------|
| `matches.csv` | [Kaggle IPL 2008–2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) | Match-level results (~816 rows) |
| `deliveries.csv` | Same dataset | Ball-by-ball data (~179k rows) |

```bash
# Download via Kaggle API
python scripts/download_data.py --kaggle

# Or create sample test data
python scripts/download_data.py --sample

# Validate files
python scripts/download_data.py --validate
```

---

## 🗂️ Project Structure

```
IPL-Win-Prediction-Model/
├── src/
│   ├── models/train_model.py          # Stacked ensemble training
│   ├── features/feature_engineering.py # 60+ features pipeline
│   └── api/main.py                    # FastAPI backend
├── streamlit_app/app.py               # Streamlit dashboard
├── notebooks/                         # EDA, experiments, SHAP analysis
├── data/raw/                          # Kaggle CSVs (git-ignored)
├── data/processed/                    # Engineered features (auto-generated)
├── models/                            # Saved artifacts (auto-generated)
├── tests/test_full_pipeline.py        # Full test suite
├── scripts/download_data.py           # Dataset downloader
├── .github/workflows/ci-cd.yml        # GitHub Actions
├── Dockerfile                         # Multi-stage build
├── docker-compose.yml                 # Production stack
├── requirements.txt
├── pyproject.toml
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

---

## 🧪 Testing

```bash
pytest                                    # Run all tests
pytest --cov=src --cov-report=html        # With coverage
pytest tests/test_full_pipeline.py -v     # Specific file
```

**Coverage targets:** Feature Engineering ≥85%, API ≥90%, Overall ≥70%

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Aranya2801](https://github.com/Aranya2801)**

*If this project helped you, please give it a ⭐*

[![GitHub stars](https://img.shields.io/github/stars/Aranya2801/IPL-Win-Prediction-Model?style=social)](https://github.com/Aranya2801/IPL-Win-Prediction-Model/stargazers)

</div>

