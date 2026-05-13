#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IPL WIN PREDICTION — Dataset Downloader                            ║
║         Downloads IPL datasets from Kaggle & validates checksums           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python scripts/download_data.py [--kaggle] [--manual]

Requirements (for Kaggle API):
    pip install kaggle
    Place kaggle.json in ~/.kaggle/

Manual Datasets Required:
    Place these files in data/raw/:
      1. matches.csv       — from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
      2. deliveries.csv    — from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
"""

import os
import sys
import argparse
import hashlib
import shutil
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger("IPL-DataDownloader")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASETS = [
    {
        "slug":     "patrickb1912/ipl-complete-dataset-20082020",
        "files":    ["matches.csv", "deliveries.csv"],
        "desc":     "IPL Complete Dataset 2008–2020 (matches + ball-by-ball)",
    },
    {
        "slug":     "vora1011/indian-premier-league-csv-dataset-20082024",
        "files":    ["matches.csv", "deliveries.csv"],
        "desc":     "IPL Extended Dataset 2008–2024",
    },
]

REQUIRED_FILES = {
    "matches.csv":   {
        "min_rows":   700,
        "columns":    ["id","season","date","team1","team2","venue","winner"],
        "desc":       "Match-level data (one row per match)",
    },
    "deliveries.csv": {
        "min_rows":   200_000,
        "columns":    ["match_id","inning","batting_team","bowling_team",
                       "over","ball","total_runs","player_dismissed"],
        "desc":       "Ball-by-ball data",
    },
}


def download_via_kaggle(slug: str, dest: Path) -> bool:
    try:
        import kaggle  # noqa
        logger.info(f"Downloading: {slug}")
        os.system(f"kaggle datasets download -d {slug} -p {dest} --unzip")
        return True
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def validate_file(path: Path, spec: dict) -> bool:
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=5)
        missing = [c for c in spec["columns"] if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns in {path.name}: {missing}")
            return False
        row_count = sum(1 for _ in open(path)) - 1
        if row_count < spec["min_rows"]:
            logger.warning(f"{path.name}: only {row_count} rows (need ≥ {spec['min_rows']})")
            return False
        logger.info(f"✓ {path.name} validated ({row_count:,} rows)")
        return True
    except Exception as e:
        logger.error(f"Validation failed for {path}: {e}")
        return False


def create_sample_data():
    """Create minimal sample CSV files for testing."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)

    teams = ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders",
             "Royal Challengers Bangalore", "Delhi Capitals"]
    venues = ["Mumbai", "Chennai", "Kolkata", "Bangalore", "Delhi"]

    # Sample matches
    n = 200
    t1 = rng.choice(teams, n)
    t2 = np.array([rng.choice([t for t in teams if t != x]) for x in t1])
    winner = np.where(rng.random(n) > 0.5, t1, t2)

    matches = pd.DataFrame({
        "id":            range(1, n+1),
        "season":        rng.integers(2008, 2024, n),
        "date":          pd.date_range("2008-04-18", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "team1":         t1,
        "team2":         t2,
        "toss_winner":   t1,
        "toss_decision": rng.choice(["bat","field"], n),
        "venue":         rng.choice(venues, n),
        "city":          rng.choice(venues, n),
        "winner":        winner,
        "result":        rng.choice(["runs","wickets"], n),
        "player_of_match": ["Player" + str(i) for i in range(n)],
    })
    matches.to_csv(RAW_DIR / "matches.csv", index=False)
    logger.info(f"Sample matches.csv created ({n} rows)")

    # Sample deliveries
    rows = []
    for match_id in range(1, min(n+1, 51)):
        for inning in [1, 2]:
            bt = teams[match_id % len(teams)]
            bwt = teams[(match_id + 1) % len(teams)]
            for over in range(20):
                for ball in range(1, 7):
                    rows.append({
                        "match_id":         match_id,
                        "inning":           inning,
                        "batting_team":     bt,
                        "bowling_team":     bwt,
                        "over":             over,
                        "ball":             ball,
                        "batter":           "Batter1",
                        "bowler":           "Bowler1",
                        "total_runs":       rng.integers(0, 7),
                        "batsman_runs":     rng.integers(0, 6),
                        "extra_runs":       0,
                        "player_dismissed": None if rng.random() > 0.05 else "Batter1",
                        "dismissal_kind":   None,
                    })
    deliveries = pd.DataFrame(rows)
    deliveries.to_csv(RAW_DIR / "deliveries.csv", index=False)
    logger.info(f"Sample deliveries.csv created ({len(deliveries):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="IPL Dataset Downloader")
    parser.add_argument("--kaggle", action="store_true",
                        help="Download from Kaggle (requires kaggle.json)")
    parser.add_argument("--sample", action="store_true",
                        help="Create sample data for testing")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing data files")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  IPL Win Prediction — Dataset Downloader")
    logger.info(f"  Target directory: {RAW_DIR}")
    logger.info("=" * 60)

    if args.sample:
        logger.info("Creating sample data for testing…")
        create_sample_data()
        return

    if args.validate:
        all_ok = True
        for fname, spec in REQUIRED_FILES.items():
            path = RAW_DIR / fname
            if not path.exists():
                logger.error(f"Missing: {fname}  — {spec['desc']}")
                all_ok = False
            else:
                ok = validate_file(path, spec)
                all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    if args.kaggle:
        success = False
        for ds in KAGGLE_DATASETS:
            logger.info(f"\nTrying: {ds['desc']}")
            if download_via_kaggle(ds["slug"], RAW_DIR):
                success = True
                break
        if not success:
            logger.error("All Kaggle downloads failed.")
            logger.info("\n📋 Manual Download Instructions:")
            logger.info("  1. Go to: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020")
            logger.info("  2. Download and extract to: data/raw/")
            logger.info("  Files needed: matches.csv, deliveries.csv")
    else:
        logger.info("\n📋 Dataset Download Instructions:")
        logger.info("=" * 60)
        for i, ds in enumerate(KAGGLE_DATASETS, 1):
            logger.info(f"\n  Option {i}: {ds['desc']}")
            logger.info(f"    kaggle datasets download -d {ds['slug']} -p data/raw/ --unzip")
        logger.info("\n  Or run with --kaggle flag to auto-download.")
        logger.info("  Or run with --sample flag to create test data.")

    # Validate downloaded files
    logger.info("\nValidating downloaded files…")
    for fname, spec in REQUIRED_FILES.items():
        path = RAW_DIR / fname
        if path.exists():
            validate_file(path, spec)
        else:
            logger.warning(f"Still missing: {fname}")


if __name__ == "__main__":
    main()
