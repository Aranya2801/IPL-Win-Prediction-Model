# Changelog

All notable changes to IPL Win Prediction Model are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2024-01-15

### Added
- **Stacked Ensemble** — XGBoost + LightGBM + RandomForest + GradientBoosting with Logistic Regression meta-learner
- **Bayesian Hyperparameter Optimization** — Optuna with TPE sampler (150 trials per model)
- **SHAP Explainability** — TreeExplainer integrated into training and API response
- **FastAPI Backend** — Async REST API with rate limiting, caching, and health checks
- **Streamlit Dashboard** — Real-time Plotly gauges, win probability timeline, pressure chart
- **Docker Compose** — Full production stack (API + Streamlit + Redis + MLflow + Nginx + Prometheus + Grafana)
- **GitHub Actions CI/CD** — Lint, test, train, Docker build/push pipeline
- **60+ Engineered Features** — Pressure index, momentum score, H2H rates, venue analytics, phase stats
- **10-Fold Stratified CV** — Robust evaluation with ROC-AUC and accuracy reporting
- **MLflow Tracking** — Experiment metadata and artifact logging
- **MIT License**

### Changed
- Upgraded from single XGBoost to full stacked ensemble
- Feature set expanded from 8 → 25 features

---

## [1.0.0] — 2023-09-01

### Added
- Basic XGBoost model
- Flask backend
- Streamlit frontend
- Kaggle IPL dataset integration
- Initial README
