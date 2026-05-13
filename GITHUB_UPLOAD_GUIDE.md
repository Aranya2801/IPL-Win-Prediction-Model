# 🚀 GitHub Upload Guide — IPL Win Prediction Model

Follow these exact steps to upload everything to your GitHub repo.

---

## Step 1 — Download the ZIP
Download `IPL-Win-Prediction-Model.zip` from Claude (the file shared below).

---

## Step 2 — Extract the ZIP
```
Unzip → you get a folder called: IPL-Win-Prediction-Model/
```

---

## Step 3 — Clone YOUR existing GitHub repo locally
```bash
git clone https://github.com/Aranya2801/IPL-Win-Prediction-Model.git
cd IPL-Win-Prediction-Model
```

---

## Step 4 — Copy all extracted files into the cloned folder
Copy everything from the extracted ZIP folder INTO the cloned repo folder.
Replace any existing files (like the old README.md).

---

## Step 5 — Get your IPL Dataset from Kaggle
You need 2 files in `data/raw/`:

Option A — Kaggle API (easiest):
```bash
pip install kaggle
# Place your kaggle.json at: ~/.kaggle/kaggle.json
python scripts/download_data.py --kaggle
```

Option B — Manual download:
1. Go to: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
2. Download and extract
3. Copy `matches.csv` and `deliveries.csv` → `data/raw/`

Option C — Use sample data (for testing only):
```bash
python scripts/download_data.py --sample
```

---

## Step 6 — Commit and push EVERYTHING
```bash
git add .
git commit -m "🏏 v2.0.0 — MIT-level stacked ensemble, FastAPI, Streamlit, Docker, CI/CD"
git push origin main
```

---

## Step 7 — Add Topics to your GitHub repo
Go to your repo → Settings → Topics, add:
```
machine-learning  ipl  cricket  xgboost  lightgbm  fastapi
streamlit  docker  python  shap  optuna  ensemble-learning
```

---

## Step 8 — Enable GitHub Pages (optional)
Repo Settings → Pages → Deploy from branch `main` → `/docs`

---

## Step 9 — Add repo description
Go to your repo → click the gear ⚙️ next to "About":
```
🏏 Real-time IPL win probability engine — Stacked Ensemble ML (XGBoost+LightGBM+RF+GB),
Bayesian optimization, SHAP explainability, FastAPI, Streamlit, Docker, CI/CD
```

---

## IMPORTANT — What NOT to commit
The .gitignore already handles these, but double-check:
- ❌ data/raw/*.csv  (too large, private)
- ❌ models/*.pkl   (too large, auto-generated)
- ❌ kaggle.json    (your secret API key)
- ❌ .env files     (secrets)

---

## Dataset Links to mention in your repo description

| Dataset | URL |
|---------|-----|
| IPL 2008–2020 (Primary) | https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020 |
| IPL 2008–2024 (Extended) | https://www.kaggle.com/datasets/vora1011/indian-premier-league-csv-dataset-20082024 |

Add these links to your README under the Dataset section.

---

## Your repo will have:
- 40 files
- 4,200+ lines of code
- MIT License ✅
- CI/CD badge ✅
- Full Docker stack ✅
- Stunning README ✅
