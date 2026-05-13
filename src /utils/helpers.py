"""
IPL Win Prediction — Utility Helpers
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("IPL-Utils")

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


def timer(fn):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        logger.debug(f"{fn.__name__} completed in {(time.time()-t0)*1000:.1f}ms")
        return result
    return wrapper


def compute_required_run_rate(runs_left: int, balls_left: int) -> float:
    """Calculate Required Run Rate."""
    if balls_left <= 0:
        return 99.99
    return round(runs_left / (balls_left / 6), 4)


def compute_pressure_index(runs_left: int, balls_left: int, wickets_fallen: int) -> float:
    """
    Pressure Index — original IPL-WP formula:
        PI = (runs_left / balls_left) * (1 + wickets_fallen / 10)
    """
    return round((runs_left / (balls_left + 1e-6)) * (1 + wickets_fallen / 10), 4)


def overs_to_balls(overs: float) -> int:
    """Convert fractional overs (e.g. 10.3) to ball count."""
    overs_int  = int(overs)
    balls_part = round((overs % 1) * 10)
    return overs_int * 6 + balls_part


def balls_to_overs(balls: int) -> str:
    """Convert ball count to human-readable overs string."""
    return f"{balls // 6}.{balls % 6}"


def confidence_label(prob: float) -> str:
    """Return confidence label based on distance from 50%."""
    gap = abs(prob - 0.5)
    if gap > 0.35: return "Very High"
    if gap > 0.20: return "High"
    if gap > 0.10: return "Medium"
    return "Low"


def validate_match_state(state: Dict[str, Any]) -> Optional[str]:
    """Validate match state dict. Returns error message or None."""
    required = ["batting_team", "bowling_team", "city",
                "target", "current_score", "wickets_fallen", "overs_completed"]
    for key in required:
        if key not in state:
            return f"Missing field: {key}"

    if state["batting_team"] == state["bowling_team"]:
        return "batting_team and bowling_team must be different"
    if state["current_score"] > state["target"] + 10:
        return "current_score seems too high relative to target"
    if state["wickets_fallen"] > 10:
        return "wickets_fallen cannot exceed 10"
    if state["overs_completed"] > 20:
        return "overs_completed cannot exceed 20"
    balls_frac = round((state["overs_completed"] % 1) * 10)
    if balls_frac >= 6:
        return "Fractional part of overs_completed must be < 0.6"
    return None


def flatten_json(nested: Dict, sep: str = ".") -> Dict:
    """Flatten a nested dictionary."""
    out = {}
    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}{sep}" if prefix else f"{k}{sep}")
        else:
            out[prefix.rstrip(sep)] = obj
    _flatten(nested)
    return out


def load_model_config(config_path: Path) -> Dict:
    """Load model config JSON."""
    with open(config_path) as f:
        return json.load(f)
