"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            IPL WIN PREDICTION — Advanced Model Training Pipeline            ║
║            MIT-Level Ensemble Learning with Bayesian Optimization           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author: Aranya2801
Version: 2.0.0
"""

import os
import sys
import json
import time
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import mlflow
import mlflow.sklearn
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IPL-WinPredictor")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 10
OPTUNA_TRIALS = 150
TARGET_COL = "win"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_name: str
    version: str = "2.0.0"
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict = field(default_factory=dict)
    cv_score: float = 0.0
    auc_score: float = 0.0
    log_loss_score: float = 0.0
    trained_at: str = ""
    shap_summary: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class IPLDataLoader:
    """Loads and validates IPL match & ball-by-ball data."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        matches_path = self.data_dir / "matches_engineered.csv"
        bbb_path     = self.data_dir / "ball_by_ball_engineered.csv"

        if not matches_path.exists():
            logger.warning("Processed data not found. Running feature engineering first...")
            from src.features.feature_engineering import IPLFeatureEngineer
            IPLFeatureEngineer().run()

        matches = pd.read_csv(matches_path, parse_dates=["date"])
        bbb     = pd.read_csv(bbb_path)
        logger.info(f"Loaded matches: {matches.shape}, ball-by-ball: {bbb.shape}")
        self._validate(matches)
        return matches, bbb

    def _validate(self, df: pd.DataFrame):
        required = [TARGET_COL, "batting_team", "bowling_team",
                    "city", "runs_left", "balls_left", "wickets_left",
                    "current_run_rate", "required_run_rate",
                    "target", "total_runs_x"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        logger.info("Data validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
class BayesianOptimizer:
    """Optuna-based Bayesian optimization for each model."""

    def __init__(self, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS):
        self.X = X
        self.y = y
        self.cv = cv
        self.skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    def _score(self, model) -> float:
        scores = cross_val_score(
            model, self.X, self.y,
            cv=self.skf, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    def optimize_xgboost(self, n_trials: int = OPTUNA_TRIALS) -> Dict:
        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
                "max_depth":         trial.suggest_int("max_depth", 3, 12),
                "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                "gamma":             trial.suggest_float("gamma", 0, 5),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
                "random_state":      RANDOM_STATE,
                "use_label_encoder": False,
                "eval_metric":       "logloss",
                "n_jobs":            -1,
            }
            model = xgb.XGBClassifier(**params)
            return self._score(model)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        logger.info(f"XGBoost best AUC: {study.best_value:.4f}")
        return study.best_params

    def optimize_lightgbm(self, n_trials: int = OPTUNA_TRIALS) -> Dict:
        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
                "max_depth":         trial.suggest_int("max_depth", 3, 15),
                "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
                "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
                "random_state":      RANDOM_STATE,
                "verbose":           -1,
                "n_jobs":            -1,
            }
            model = lgb.LGBMClassifier(**params)
            return self._score(model)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        logger.info(f"LightGBM best AUC: {study.best_value:.4f}")
        return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class IPLEnsembleModel:
    """
    Stacked ensemble:
      Level-0 : XGBoost + LightGBM + RandomForest + GradientBoosting
      Level-1 : Calibrated Logistic Regression meta-learner
    """

    def __init__(self, feature_cols: List[str], optimize: bool = True):
        self.feature_cols = feature_cols
        self.optimize = optimize
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.model: Optional[StackingClassifier] = None
        self.config = ModelConfig(model_name="IPL-StackedEnsemble")
        self.shap_explainer = None

    # ── preprocessing ────────────────────────────────────────────────────────
    def _encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0]
                        if x in le.classes_ else -1
                    )
        return df

    def _prepare(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = self._encode(df[self.feature_cols], fit=fit)
        if fit:
            return self.scaler.fit_transform(df.values)
        return self.scaler.transform(df.values)

    # ── training ─────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        logger.info("Starting model training pipeline…")
        X_raw = df[self.feature_cols]
        y     = df[TARGET_COL].values
        X     = self._prepare(X_raw, fit=True)

        # Bayesian optimization
        if self.optimize:
            logger.info("Running Bayesian hyperparameter optimization…")
            optimizer = BayesianOptimizer(X, y)
            xgb_params  = optimizer.optimize_xgboost(n_trials=OPTUNA_TRIALS)
            lgbm_params = optimizer.optimize_lightgbm(n_trials=OPTUNA_TRIALS)
        else:
            xgb_params  = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                           "random_state": RANDOM_STATE, "use_label_encoder": False,
                           "eval_metric": "logloss"}
            lgbm_params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                           "random_state": RANDOM_STATE, "verbose": -1}

        # Base learners
        base_learners = [
            ("xgb",  xgb.XGBClassifier(**xgb_params, n_jobs=-1)),
            ("lgbm", lgb.LGBMClassifier(**lgbm_params, n_jobs=-1)),
            ("rf",   RandomForestClassifier(n_estimators=500, max_depth=12,
                                            random_state=RANDOM_STATE, n_jobs=-1)),
            ("gb",   GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                learning_rate=0.05,
                                                random_state=RANDOM_STATE)),
        ]

        # Meta-learner
        meta = CalibratedClassifierCV(
            LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE),
            method="isotonic", cv=5
        )

        # Stacking ensemble
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE),
            passthrough=True,
            n_jobs=-1,
        )

        logger.info("Fitting stacked ensemble…")
        t0 = time.time()
        self.model.fit(X, y)
        logger.info(f"Training complete in {time.time()-t0:.1f}s")

        # Cross-validation evaluation
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_auc  = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc",  n_jobs=-1)
        cv_acc  = cross_val_score(self.model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

        self.config.cv_score  = float(cv_acc.mean())
        self.config.auc_score = float(cv_auc.mean())
        logger.info(f"CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        logger.info(f"CV ROC-AUC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        # SHAP explainability (on XGBoost base learner)
        self._compute_shap(X, X_raw)

        # MLflow logging
        self._log_mlflow(cv_acc, cv_auc)

        return self

    # ── SHAP ─────────────────────────────────────────────────────────────────
    def _compute_shap(self, X: np.ndarray, X_raw: pd.DataFrame):
        logger.info("Computing SHAP values…")
        xgb_model = self.model.named_estimators_["xgb"]
        self.shap_explainer = shap.TreeExplainer(xgb_model)
        shap_vals = self.shap_explainer.shap_values(X[:500])  # sample
        feature_importance = np.abs(shap_vals).mean(axis=0)
        self.config.shap_summary = {
            col: float(imp)
            for col, imp in zip(self.feature_cols, feature_importance)
        }
        logger.info("SHAP values computed ✓")

    # ── MLflow ───────────────────────────────────────────────────────────────
    def _log_mlflow(self, cv_acc, cv_auc):
        try:
            with mlflow.start_run(run_name="IPL-StackedEnsemble-v2"):
                mlflow.log_param("cv_folds",      CV_FOLDS)
                mlflow.log_param("optuna_trials",  OPTUNA_TRIALS)
                mlflow.log_param("n_features",     len(self.feature_cols))
                mlflow.log_metric("cv_accuracy_mean", float(cv_acc.mean()))
                mlflow.log_metric("cv_accuracy_std",  float(cv_acc.std()))
                mlflow.log_metric("cv_auc_mean",      float(cv_auc.mean()))
                mlflow.log_metric("cv_auc_std",       float(cv_auc.std()))
                mlflow.sklearn.log_model(self.model, "stacked_ensemble")
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df[self.feature_cols], fit=False)
        return self.model.predict_proba(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df).argmax(axis=1)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None):
        path = path or MODEL_DIR / "ipl_stacked_ensemble.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # Save config
        config_path = path.with_suffix(".json")
        self.config.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Model saved → {path}")
        logger.info(f"Config saved → {config_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IPLEnsembleModel":
        path = path or MODEL_DIR / "ipl_stacked_ensemble.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "batting_team", "bowling_team", "city",
    "runs_left", "balls_left", "wickets_left",
    "target", "current_run_rate", "required_run_rate",
    "total_runs_x", "avg_powerplay_runs",
    "batting_team_win_rate", "bowling_team_win_rate",
    "batting_team_avg_score", "bowling_team_avg_score",
    "venue_avg_score", "venue_win_batting_first_rate",
    "h2h_batting_win_rate", "last5_batting_win_rate",
    "last5_bowling_win_rate", "wicket_phase",
    "run_rate_diff", "pressure_index",
    "momentum_score", "innings",
]


def main():
    loader = IPLDataLoader()
    df, _ = loader.load()

    model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=True)
    model.train(df)
    model.save()

    logger.info("=" * 60)
    logger.info("  Training pipeline complete!")
    logger.info(f"  CV Accuracy : {model.config.cv_score:.4f}")
    logger.info(f"  CV AUC      : {model.config.auc_score:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
