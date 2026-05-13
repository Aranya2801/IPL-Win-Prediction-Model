"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IPL WIN PREDICTION — Production FastAPI Backend                    ║
║         RESTful API with async endpoints, rate limiting & caching          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import time
import logging
import hashlib
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import aiofiles

logger = logging.getLogger("IPL-API")
ROOT_DIR   = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "ipl_stacked_ensemble.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class MatchState(BaseModel):
    """Live match state for real-time win probability prediction."""
    batting_team:           str   = Field(..., example="Mumbai Indians")
    bowling_team:           str   = Field(..., example="Chennai Super Kings")
    city:                   str   = Field(..., example="Mumbai")
    target:                 int   = Field(..., ge=1, le=400, example=185)
    current_score:          int   = Field(..., ge=0, le=400, example=87)
    wickets_fallen:         int   = Field(..., ge=0, le=10, example=3)
    overs_completed:        float = Field(..., ge=0, le=20, example=10.3)
    batting_team_win_rate:  float = Field(0.5, ge=0, le=1)
    bowling_team_win_rate:  float = Field(0.5, ge=0, le=1)
    h2h_batting_win_rate:   float = Field(0.5, ge=0, le=1)
    venue_win_batting_first_rate: float = Field(0.5, ge=0, le=1)

    @validator("overs_completed")
    def validate_overs(cls, v):
        balls_in_over = round((v % 1) * 10)
        if balls_in_over >= 6:
            raise ValueError("Overs fraction cannot be ≥ 0.6")
        return v

    class Config:
        schema_extra = {
            "example": {
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
                "venue_win_batting_first_rate": 0.48,
            }
        }


class PredictionResponse(BaseModel):
    batting_team:          str
    bowling_team:          str
    batting_win_prob:      float
    bowling_win_prob:      float
    predicted_winner:      str
    confidence:            str
    runs_left:             int
    balls_left:            int
    wickets_left:          int
    current_run_rate:      float
    required_run_rate:     float
    run_rate_diff:         float
    pressure_index:        float
    top_features:          Dict[str, float]
    model_version:         str
    inference_time_ms:     float
    cache_hit:             bool


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    uptime_s:      float
    requests:      int


class TeamStatsRequest(BaseModel):
    team: str
    season: Optional[int] = None
    last_n: int = Field(10, ge=1, le=50)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    _instance: Optional["ModelStore"] = None
    model = None
    config: Dict = {}
    load_time: float = 0.0
    request_count: int = 0
    start_time: float = time.time()

    @classmethod
    def get(cls) -> "ModelStore":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        try:
            from src.models.train_model import IPLEnsembleModel
            self.model = IPLEnsembleModel.load(MODEL_PATH)
            config_path = MODEL_PATH.with_suffix(".json")
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
            self.load_time = time.time()
            logger.info("Model loaded successfully ✓")
        except FileNotFoundError:
            logger.warning("Model file not found. Run training first.")
            self.model = None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def state_to_features(state: MatchState) -> pd.DataFrame:
    """Convert MatchState to feature DataFrame."""
    overs_int  = int(state.overs_completed)
    balls_this = round((state.overs_completed % 1) * 10)
    balls_done = overs_int * 6 + balls_this
    balls_left = max(0, 120 - balls_done)
    runs_left  = max(0, state.target - state.current_score)
    wickets_left = max(0, 10 - state.wickets_fallen)
    crr = (state.current_score / (balls_done / 6)) if balls_done > 0 else 0
    rrr = (runs_left / (balls_left / 6)) if balls_left > 0 else 99.0
    rrd = crr - rrr
    pressure = (runs_left / (balls_left + 1e-6)) * (1 + state.wickets_fallen / 10)

    return pd.DataFrame([{
        "batting_team":                   state.batting_team,
        "bowling_team":                   state.bowling_team,
        "city":                           state.city,
        "runs_left":                      runs_left,
        "balls_left":                     balls_left,
        "wickets_left":                   wickets_left,
        "target":                         state.target,
        "current_run_rate":               round(crr, 4),
        "required_run_rate":              round(rrr, 4),
        "total_runs_x":                   state.current_score,
        "avg_powerplay_runs":             42.0,   # league avg default
        "batting_team_win_rate":          state.batting_team_win_rate,
        "bowling_team_win_rate":          state.bowling_team_win_rate,
        "batting_team_avg_score":         165.0,  # reasonable default
        "bowling_team_avg_score":         163.0,
        "venue_avg_score":                168.0,
        "venue_win_batting_first_rate":   state.venue_win_batting_first_rate,
        "h2h_batting_win_rate":           state.h2h_batting_win_rate,
        "last5_batting_win_rate":         state.batting_team_win_rate,
        "last5_bowling_win_rate":         state.bowling_team_win_rate,
        "wicket_phase":                   state.wickets_fallen,
        "run_rate_diff":                  round(rrd, 4),
        "pressure_index":                 round(pressure, 4),
        "momentum_score":                 round(rrd * 0.5, 4),
        "innings":                        2,
    }])


def cache_key(state: MatchState) -> str:
    raw = json.dumps(state.dict(), sort_keys=True)
    return f"ipl:{hashlib.md5(raw.encode()).hexdigest()}"


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    ModelStore.get()
    logger.info("IPL Prediction API started ✓")
    yield
    # shutdown
    logger.info("IPL Prediction API shutting down…")


app = FastAPI(
    title="IPL Win Prediction API",
    description=(
        "Production-grade REST API for real-time IPL match win probability "
        "predictions using a Stacked Ensemble ML model (XGBoost + LightGBM + "
        "RandomForest + GradientBoosting) with Bayesian-optimised hyperparameters."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "IPL Win Prediction API v2.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    store = ModelStore.get()
    return HealthResponse(
        status="healthy" if store.model else "degraded",
        model_loaded=store.model is not None,
        model_version=store.config.get("version", "unknown"),
        uptime_s=round(time.time() - store.start_time, 1),
        requests=store.request_count,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("60/minute")
async def predict(request: Request, state: MatchState):
    """
    Predict win probability given live match state.
    Returns calibrated probabilities for both teams with SHAP feature contributions.
    """
    store = ModelStore.get()
    store.request_count += 1

    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    t0 = time.time()
    ck = cache_key(state)
    cache_hit = False

    # In-memory LRU cache (Redis optional)
    cached = _cache_get(ck)
    if cached:
        cached["cache_hit"] = True
        cached["inference_time_ms"] = round((time.time() - t0) * 1000, 2)
        return cached

    # Inference
    features = state_to_features(state)
    proba = store.model.predict_proba(features)[0]
    batting_prob = float(proba[1])
    bowling_prob = float(proba[0])

    runs_left  = int(features["runs_left"].iloc[0])
    balls_left = int(features["balls_left"].iloc[0])
    wickets_left = int(features["wickets_left"].iloc[0])
    crr = float(features["current_run_rate"].iloc[0])
    rrr = float(features["required_run_rate"].iloc[0])
    rrd = float(features["run_rate_diff"].iloc[0])
    pi  = float(features["pressure_index"].iloc[0])

    # Top 5 SHAP features (from stored summary)
    shap_summary = store.config.get("shap_summary", {})
    top_features = dict(sorted(shap_summary.items(),
                               key=lambda x: x[1], reverse=True)[:5])

    confidence = (
        "Very High" if abs(batting_prob - 0.5) > 0.35 else
        "High"      if abs(batting_prob - 0.5) > 0.20 else
        "Medium"    if abs(batting_prob - 0.5) > 0.10 else
        "Low"
    )

    response = {
        "batting_team":      state.batting_team,
        "bowling_team":      state.bowling_team,
        "batting_win_prob":  round(batting_prob, 4),
        "bowling_win_prob":  round(bowling_prob, 4),
        "predicted_winner":  state.batting_team if batting_prob >= 0.5 else state.bowling_team,
        "confidence":        confidence,
        "runs_left":         runs_left,
        "balls_left":        balls_left,
        "wickets_left":      wickets_left,
        "current_run_rate":  round(crr, 2),
        "required_run_rate": round(rrr, 2),
        "run_rate_diff":     round(rrd, 2),
        "pressure_index":    round(pi, 2),
        "top_features":      top_features,
        "model_version":     store.config.get("version", "2.0.0"),
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
        "cache_hit":         False,
    }
    _cache_set(ck, response)
    return response


@app.post("/predict/batch", tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_batch(request: Request, states: List[MatchState]):
    """Batch prediction for multiple match states (max 50)."""
    if len(states) > 50:
        raise HTTPException(status_code=400, detail="Max 50 states per batch")
    results = []
    for state in states:
        inner_req = Request(scope=request.scope)
        result = await predict(inner_req, state)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/teams", tags=["Data"])
async def list_teams():
    """List all IPL teams in the model."""
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Kolkata Knight Riders",
        "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
        "Royal Challengers Bangalore", "Sunrisers Hyderabad",
        "Gujarat Titans", "Lucknow Super Giants",
    ]
    return {"teams": sorted(teams), "count": len(teams)}


@app.get("/venues", tags=["Data"])
async def list_venues():
    """List all IPL venues in the model."""
    venues = [
        "Mumbai", "Chennai", "Kolkata", "Bangalore", "Delhi",
        "Hyderabad", "Jaipur", "Mohali", "Pune", "Ahmedabad",
        "Indore", "Cuttack", "Ranchi", "Dharamsala",
    ]
    return {"venues": sorted(venues), "count": len(venues)}


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return model metadata and performance metrics."""
    store = ModelStore.get()
    return {
        "config": store.config,
        "model_path": str(MODEL_PATH),
        "loaded": store.model is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE IN-MEMORY CACHE (replace with Redis in production)
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: Dict = {}
_CACHE_TTL = 300  # 5 minutes

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    return None

def _cache_set(key: str, val):
    _CACHE[key] = {"val": val, "ts": time.time()}
    if len(_CACHE) > 10_000:
        oldest = min(_CACHE, key=lambda k: _CACHE[k]["ts"])
        del _CACHE[oldest]


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000,
                reload=True, log_level="info")
