# Snaphomz Trial – Data Science & AI/LLM Integration

## Setup
- Python version: 3.9+ (tested on macOS, CPU/MPS)
- OS / CPU / GPU: macOS (Apple Silicon or Intel); CPU-only is fine
- Create env and install deps:
  ```
  pip3 install -r requirements.txt
  ```
- Local model setup:
  - I use the LLM feature purely TF‑IDF (no external models)

## How to Run
1) Open the notebook and run all cells in order:
   - `notebooks/trial.ipynb`
2) Key sections inside the notebook:
   - EDA → Feature Engineering → Model Train → LLM Integration (RAG mini‑Q&A)

## Decisions
- Target & features:
  - Target: `price_per_sqft = price / sqft`
  - Features: numeric (`sqft`, `beds`, `baths`, ratios, logs), categorical (`city` one‑hot), text‑derived flags (`has_pool`, `has_garage`, `is_quiet`, `is_updated`, `has_backyard`), city priors (`city_pps_median/mean/count`).
- Train/test split & metrics:
  - Split: 80/20
  - Regression metrics: MAE / RMSE / R²
  - Classification metrics (optional banding): precision / recall / F1, confusion matrix
- Retrieval / summary approach (LLM feature):
  - Local TF‑IDF vectors (`TfidfVectorizer`) + `NearestNeighbors` (cosine) for top‑k remark retrieval
  - Lightweight rule‑based answer synthesis (no external API keys). Returns answer + supporting contexts
- Assumptions & trade‑offs:
  - Small dataset → prefer robust stats (medians, logs) and simple baselines for clarity
  - City fixed‑effect captured via one‑hot and city median pps prior
  - Text understanding is minimal (keyword flags + TF‑IDF); easy to upgrade to open embedding models

## Results (TL;DR)
- Key EDA insights:
  - `price` and `sqft` are right‑skewed; use medians/logs and IQR for outliers
  - `price_per_sqft` varies materially across cities; medians are more stable than means
  - `price` strongly correlates with `sqft`; multicollinearity among `beds/baths/sqft`
- Best regression (test set):
  - LinearRegression → MAE: 111.4, RMSE: 22746.9, R²: 0.584
  - RandomForestRegressor → MAE: 107.9, RMSE: 21361.2, R²: 0.610 (best)
  - Takeaway: size + city effects + light text features yield moderate signal (R² ≈ 0.61); non‑linear model helps
- Optional classification (pps bands):
  - LogisticRegression accuracy ≈ 0.72; macro‑F1 ≈ 0.60 (med class recall lower)
- LLM feature (RAG mini‑Q&A):
  - TF‑IDF retriever over `remarks` + rule‑based synthesis; returns answer and supporting text spans

## Repository Layout
- `data/` – input CSV (`listings_sample.csv`)
- `notebooks/trial.ipynb` – main notebook (EDA, features, modeling, LLM)
- `templates/README.template.md` – original template
- `tasks/trial_brief.md` – original assignment brief
- `results/one_pager.pdf` – outcome for this project(other things in the results folder you can just ignore)


## Known Limitations
- Small dataset; outlier sensitivity and variance in metrics
- Simple baselines; limited hyperparameter tuning
- Text understanding is minimal; no heavy LLMs by default (fully local)
- City effects absorbed via one‑hot + priors; no spatial coordinates

## Upgrade Paths
- Swap TF‑IDF with open embeddings (e.g., `BAAI/bge-small-en-v1.5` / `bge-m3`) and optional reranker (`BAAI/bge-reranker-base`)
- Add local generation (e.g., `google/flan-t5-small`) for richer answers or auto‑summaries
- Tune models (RF depth/trees), add interactions, or per‑city models

## Time Spent
- Approx hours: 2 hours

