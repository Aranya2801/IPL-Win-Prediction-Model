"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IPL WIN PREDICTION — Comprehensive Test Suite                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_match_state():
    return {
        "batting_team":                 "Mumbai Indians",
        "bowling_team":                 "Chennai Super Kings",
        "city":                         "Mumbai",
        "target":                       185,
        "current_score":                87,
        "wickets_fallen":               3,
        "overs_completed":              10.3,
        "batting_team_win_rate":        0.62,
        "bowling_team_win_rate":        0.58,
        "h2h_batting_win_rate":         0.55,
        "venue_win_batting_first_rate": 0.48,
    }


@pytest.fixture
def sample_dataframe():
    """Minimal valid DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "batting_team":                 rng.choice(["MI", "CSK", "KKR"], n),
        "bowling_team":                 rng.choice(["RCB", "DC", "SRH"], n),
        "city":                         rng.choice(["Mumbai", "Chennai", "Delhi"], n),
        "runs_left":                    rng.integers(0, 150, n),
        "balls_left":                   rng.integers(0, 120, n),
        "wickets_left":                 rng.integers(0, 10, n),
        "target":                       rng.integers(100, 250, n),
        "current_run_rate":             rng.uniform(4, 14, n),
        "required_run_rate":            rng.uniform(4, 20, n),
        "total_runs_x":                 rng.integers(20, 200, n),
        "avg_powerplay_runs":           rng.uniform(35, 60, n),
        "batting_team_win_rate":        rng.uniform(0.3, 0.7, n),
        "bowling_team_win_rate":        rng.uniform(0.3, 0.7, n),
        "batting_team_avg_score":       rng.uniform(150, 185, n),
        "bowling_team_avg_score":       rng.uniform(148, 180, n),
        "venue_avg_score":              rng.uniform(155, 175, n),
        "venue_win_batting_first_rate": rng.uniform(0.4, 0.6, n),
        "h2h_batting_win_rate":         rng.uniform(0.3, 0.7, n),
        "last5_batting_win_rate":       rng.uniform(0.3, 0.7, n),
        "last5_bowling_win_rate":       rng.uniform(0.3, 0.7, n),
        "wicket_phase":                 rng.integers(0, 5, n),
        "run_rate_diff":                rng.uniform(-5, 5, n),
        "pressure_index":               rng.uniform(0, 10, n),
        "momentum_score":               rng.uniform(-3, 3, n),
        "innings":                      np.ones(n, dtype=int) * 2,
        "win":                          rng.integers(0, 2, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestFeatureEngineering:

    def test_state_to_features_basic(self, sample_match_state):
        """state_to_features should return a single-row DataFrame."""
        from src.api.main import state_to_features
        from src.api.main import MatchState
        state = MatchState(**sample_match_state)
        df = state_to_features(state)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_balls_left_calculation(self, sample_match_state):
        from src.api.main import state_to_features, MatchState
        state = MatchState(**{**sample_match_state, "overs_completed": 10.3})
        df = state_to_features(state)
        # 10.3 = 10 overs + 3 balls = 63 balls done → 57 left
        assert df["balls_left"].iloc[0] == 57

    def test_runs_left_calculation(self, sample_match_state):
        from src.api.main import state_to_features, MatchState
        state = MatchState(**{**sample_match_state,
                               "target": 185, "current_score": 87})
        df = state_to_features(state)
        assert df["runs_left"].iloc[0] == 98

    def test_no_negative_values(self, sample_match_state):
        from src.api.main import state_to_features, MatchState
        state = MatchState(**{**sample_match_state,
                               "target": 100, "current_score": 105})  # score > target
        df = state_to_features(state)
        assert df["runs_left"].iloc[0] == 0

    def test_pressure_index_positive(self, sample_match_state):
        from src.api.main import state_to_features, MatchState
        state = MatchState(**sample_match_state)
        df = state_to_features(state)
        assert df["pressure_index"].iloc[0] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestEnsembleModel:

    def test_model_instantiation(self):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        assert model.feature_cols == FEATURE_COLS
        assert model.model is None

    def test_training_pipeline(self, sample_dataframe):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        assert model.model is not None

    def test_predict_proba_shape(self, sample_dataframe):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        proba = model.predict_proba(sample_dataframe.head(10))
        assert proba.shape == (10, 2)

    def test_probabilities_sum_to_one(self, sample_dataframe):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        proba = model.predict_proba(sample_dataframe.head(20))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_range(self, sample_dataframe):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        proba = model.predict_proba(sample_dataframe.head(20))
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_cv_score_reasonable(self, sample_dataframe):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        # On random data, accuracy should be > 0.4 (not completely broken)
        assert model.config.cv_score >= 0.4

    def test_save_load_roundtrip(self, sample_dataframe, tmp_path):
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        path = tmp_path / "test_model.pkl"
        model.save(path)
        loaded = IPLEnsembleModel.load(path)
        proba_orig = model.predict_proba(sample_dataframe.head(5))
        proba_load = loaded.predict_proba(sample_dataframe.head(5))
        np.testing.assert_allclose(proba_orig, proba_load, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# API TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestAPIEndpoints:

    @pytest.fixture
    def client(self, sample_dataframe):
        """Create a test client with a mock model."""
        from src.api.main import app, ModelStore
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS

        # Train a tiny model for testing
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        store = ModelStore.get()
        store.model = model
        store.config = {"version": "test", "shap_summary": {"runs_left": 0.5}}
        return TestClient(app)

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert "model_loaded" in data

    def test_predict_endpoint_success(self, client, sample_match_state):
        r = client.post("/predict", json=sample_match_state)
        assert r.status_code == 200
        data = r.json()
        assert "batting_win_prob" in data
        assert "bowling_win_prob" in data
        assert 0 <= data["batting_win_prob"] <= 1
        assert 0 <= data["bowling_win_prob"] <= 1
        assert abs(data["batting_win_prob"] + data["bowling_win_prob"] - 1.0) < 0.01

    def test_predict_winner_field(self, client, sample_match_state):
        r = client.post("/predict", json=sample_match_state)
        data = r.json()
        assert data["predicted_winner"] in [
            sample_match_state["batting_team"],
            sample_match_state["bowling_team"]
        ]

    def test_teams_endpoint(self, client):
        r = client.get("/teams")
        assert r.status_code == 200
        assert "teams" in r.json()
        assert len(r.json()["teams"]) >= 8

    def test_venues_endpoint(self, client):
        r = client.get("/venues")
        assert r.status_code == 200
        assert "venues" in r.json()

    def test_model_info_endpoint(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_invalid_overs_rejected(self, client, sample_match_state):
        bad_state = {**sample_match_state, "overs_completed": 10.7}
        r = client.post("/predict", json=bad_state)
        assert r.status_code == 422  # Unprocessable Entity

    def test_batch_predict(self, client, sample_match_state):
        states = [sample_match_state, sample_match_state]
        r = client.post("/predict/batch", json=states)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2

    def test_batch_limit_exceeded(self, client, sample_match_state):
        states = [sample_match_state] * 51
        r = client.post("/predict/batch", json=states)
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEnd:

    def test_full_pipeline(self, sample_dataframe, tmp_path):
        """Test complete train → predict pipeline."""
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS

        # Train
        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)
        model.save(tmp_path / "e2e_model.pkl")

        # Load and predict
        loaded = IPLEnsembleModel.load(tmp_path / "e2e_model.pkl")
        test_row = sample_dataframe.head(1)
        proba = loaded.predict_proba(test_row)
        assert proba.shape == (1, 2)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_extreme_situations(self, sample_dataframe):
        """Test model handles edge cases."""
        from src.models.train_model import IPLEnsembleModel, FEATURE_COLS

        model = IPLEnsembleModel(feature_cols=FEATURE_COLS, optimize=False)
        model.train(sample_dataframe)

        # Edge: nearly impossible (needs 100 off 1 ball)
        edge = sample_dataframe.head(1).copy()
        edge["runs_left"] = 100
        edge["balls_left"] = 1
        proba = model.predict_proba(edge)
        assert proba[0][1] < 0.2  # batting team should have very low chance

        # Edge: already won (0 runs needed)
        edge2 = sample_dataframe.head(1).copy()
        edge2["runs_left"] = 0
        edge2["balls_left"] = 30
        proba2 = model.predict_proba(edge2)
        assert proba2[0][1] > 0.7  # batting team should have high chance
