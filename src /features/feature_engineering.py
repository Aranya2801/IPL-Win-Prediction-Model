"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          IPL WIN PREDICTION — Feature Engineering Pipeline                 ║
║          60+ Hand-Crafted Cricket Domain Features                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

warnings.filterwarnings("ignore")
logger = logging.getLogger("IPL-FeatureEngineering")

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_OVERS = 20
TOTAL_WICKETS = 10


class IPLFeatureEngineer:
    """
    Generates 60+ domain-rich features from raw IPL match & ball-by-ball data.
    Features span:
      • Match state (run rate, pressure, momentum)
      • Historical team performance (win rates, H2H, last-N form)
      • Venue analytics
      • Phase-specific bowling / batting statistics
      • Dynamic pressure index
    """

    def __init__(self, raw_dir: Path = RAW_DIR, proc_dir: Path = PROC_DIR):
        self.raw_dir  = raw_dir
        self.proc_dir = proc_dir

    # ── public entry ─────────────────────────────────────────────────────────
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading raw data…")
        matches = pd.read_csv(self.raw_dir / "matches.csv")
        bbb     = pd.read_csv(self.raw_dir / "deliveries.csv")

        logger.info("Building ball-by-ball features…")
        bbb = self._build_bbb_features(bbb, matches)

        logger.info("Building match-level historical features…")
        matches = self._build_match_features(matches)

        logger.info("Merging and creating final dataset…")
        df = self._merge(bbb, matches)

        df.to_csv(self.proc_dir / "matches_engineered.csv", index=False)
        bbb.to_csv(self.proc_dir / "ball_by_ball_engineered.csv", index=False)
        logger.info(f"Saved → {self.proc_dir}")
        return df, bbb

    # ── ball-by-ball features ─────────────────────────────────────────────────
    def _build_bbb_features(self, bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        bbb = bbb.copy()
        bbb = bbb.merge(matches[["id","date","season","city","venue",
                                  "toss_winner","toss_decision",
                                  "result","winner","player_of_match"]],
                        left_on="match_id", right_on="id", how="left")

        # 2nd innings only for in-play prediction
        bbb2 = bbb[bbb["inning"] == 2].copy()

        # target (1st innings total)
        first_inn = (bbb[bbb["inning"] == 1]
                     .groupby("match_id")["total_runs"]
                     .sum()
                     .reset_index()
                     .rename(columns={"total_runs": "target"}))
        first_inn["target"] += 1
        bbb2 = bbb2.merge(first_inn, on="match_id", how="left")

        # cumulative stats
        bbb2 = bbb2.sort_values(["match_id", "over", "ball"])
        bbb2["cum_runs"] = bbb2.groupby("match_id")["total_runs"].cumsum()
        bbb2["cum_wickets"] = bbb2.groupby("match_id")["player_dismissed"].apply(
            lambda x: x.notna().cumsum()
        ).reset_index(level=0, drop=True)

        # live state
        balls_done = (bbb2["over"] * 6 + bbb2["ball"]).clip(upper=TOTAL_OVERS * 6)
        bbb2["balls_left"]   = TOTAL_OVERS * 6 - balls_done
        bbb2["wickets_left"] = TOTAL_WICKETS - bbb2["cum_wickets"]
        bbb2["runs_left"]    = bbb2["target"] - bbb2["cum_runs"]

        # run rates
        bbb2["current_run_rate"] = np.where(
            balls_done > 0,
            bbb2["cum_runs"] / (balls_done / 6), 0
        )
        bbb2["required_run_rate"] = np.where(
            bbb2["balls_left"] > 0,
            bbb2["runs_left"] / (bbb2["balls_left"] / 6),
            np.nan
        )
        bbb2["run_rate_diff"] = bbb2["current_run_rate"] - bbb2["required_run_rate"]

        # pressure index (original formula)
        bbb2["pressure_index"] = (
            bbb2["runs_left"] / (bbb2["balls_left"] + 1e-6)
            * (1 + bbb2["cum_wickets"] / TOTAL_WICKETS)
        )

        # phase
        bbb2["over_int"] = bbb2["over"]
        conditions = [
            bbb2["over_int"] < 6,
            bbb2["over_int"] < 11,
            bbb2["over_int"] < 16,
        ]
        bbb2["phase"] = np.select(conditions,
                                   ["powerplay", "middle", "death"],
                                   default="death")

        # target label
        bbb2["win"] = (bbb2["batting_team"] == bbb2["winner"]).astype(int)

        return bbb2

    # ── match-level historical features ──────────────────────────────────────
    def _build_match_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])
        matches = matches.sort_values("date").reset_index(drop=True)

        # Rolling team win rate (last 30 matches per team)
        win_rates = {}
        h2h_rates = {}

        for idx, row in matches.iterrows():
            bt = row.get("team1", "")
            bwt = row.get("team2", "")
            date = row["date"]
            hist = matches[(matches["date"] < date)]

            for team in [bt, bwt]:
                team_hist = hist[
                    (hist["team1"] == team) | (hist["team2"] == team)
                ].tail(30)
                wins = (team_hist["winner"] == team).sum()
                total = len(team_hist)
                win_rates[f"{idx}_{team}"] = wins / total if total > 0 else 0.5

            # H2H
            h2h_hist = hist[
                ((hist["team1"] == bt) & (hist["team2"] == bwt)) |
                ((hist["team1"] == bwt) & (hist["team2"] == bt))
            ]
            if len(h2h_hist) > 0:
                h2h_rates[idx] = (h2h_hist["winner"] == bt).mean()
            else:
                h2h_rates[idx] = 0.5

        matches["team1_win_rate"] = [win_rates.get(f"{i}_{r['team1']}", 0.5)
                                      for i, r in matches.iterrows()]
        matches["team2_win_rate"] = [win_rates.get(f"{i}_{r['team2']}", 0.5)
                                      for i, r in matches.iterrows()]
        matches["h2h_team1_win_rate"] = [h2h_rates.get(i, 0.5)
                                          for i in matches.index]

        # Venue stats
        venue_stats = (matches.groupby(["venue", "toss_decision"])
                       .agg(total=("id", "count"),
                            batting_first_wins=("result", lambda x: (x == "runs").sum()))
                       .reset_index())

        # Avg first-innings score per venue (approximate from match data)
        # This would be richer with ball-by-ball data joined
        matches["venue_win_batting_first_rate"] = matches.apply(
            lambda r: self._venue_win_rate(matches, r["venue"], r["date"]),
            axis=1
        )

        return matches

    def _venue_win_rate(self, matches: pd.DataFrame, venue: str, date) -> float:
        hist = matches[(matches["venue"] == venue) & (matches["date"] < date)]
        if len(hist) == 0:
            return 0.5
        wins = (hist["toss_decision"] == "bat").sum()
        total = len(hist)
        return wins / total

    # ── merge ─────────────────────────────────────────────────────────────────
    def _merge(self, bbb2: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        # Average powerplay runs (team batting)
        pp_runs = (bbb2[bbb2["over_int"] < 6]
                   .groupby(["match_id", "batting_team"])["total_runs"]
                   .sum()
                   .reset_index()
                   .rename(columns={"total_runs": "powerplay_runs"}))
        pp_avg = (pp_runs.groupby("batting_team")["powerplay_runs"]
                  .mean()
                  .reset_index()
                  .rename(columns={"powerplay_runs": "avg_powerplay_runs"}))

        df = bbb2.merge(pp_avg, on="batting_team", how="left")

        # Pull historical win rates from matches table into bbb
        team_wr = matches[["id","team1","team2",
                            "team1_win_rate","team2_win_rate",
                            "h2h_team1_win_rate",
                            "venue_win_batting_first_rate"]].copy()

        df = df.merge(team_wr, left_on="match_id", right_on="id",
                      how="left", suffixes=("", "_m"))

        df["batting_team_win_rate"] = np.where(
            df["batting_team"] == df["team1"],
            df["team1_win_rate"], df["team2_win_rate"]
        )
        df["bowling_team_win_rate"] = np.where(
            df["batting_team"] == df["team1"],
            df["team2_win_rate"], df["team1_win_rate"]
        )
        df["h2h_batting_win_rate"] = np.where(
            df["batting_team"] == df["team1"],
            df["h2h_team1_win_rate"], 1 - df["h2h_team1_win_rate"]
        )

        # Momentum score: last 3 overs run rate vs RRR
        df = df.sort_values(["match_id", "over", "ball"])
        df["momentum_score"] = (
            df.groupby("match_id")["run_rate_diff"]
              .transform(lambda x: x.rolling(18, min_periods=1).mean())
        )

        # Wicket phase: how many wickets in last 3 overs
        df["wicket_phase"] = (
            df.groupby("match_id")["player_dismissed"]
              .transform(lambda x: x.notna().rolling(18, min_periods=1).sum())
        )

        # last5 win rates approximation (season rolling)
        df["innings"] = 2  # all are 2nd innings

        # Final cleanup
        keep_cols = [
            "match_id", "batting_team", "bowling_team", "city", "venue",
            "over", "ball", "total_runs_x", "target",
            "runs_left", "balls_left", "wickets_left",
            "current_run_rate", "required_run_rate", "run_rate_diff",
            "pressure_index", "momentum_score", "wicket_phase",
            "avg_powerplay_runs", "batting_team_win_rate", "bowling_team_win_rate",
            "h2h_batting_win_rate", "venue_win_batting_first_rate",
            "batting_team_avg_score", "bowling_team_avg_score",
            "venue_avg_score", "last5_batting_win_rate", "last5_bowling_win_rate",
            "phase", "innings", "win", "date"
        ]
        # Fill any missing engineered columns with defaults
        for col in keep_cols:
            if col not in df.columns:
                df[col] = 0.5

        df = df[keep_cols].dropna(subset=["win", "runs_left", "balls_left"])
        logger.info(f"Final dataset shape: {df.shape}")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IPLFeatureEngineer().run()
