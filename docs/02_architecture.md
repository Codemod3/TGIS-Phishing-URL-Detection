# Architecture & Design Choices

This file explains how the project is structured and *why* it was designed that way.

---

## The Two-Layer Split: `api/` vs `src/`

The project deliberately separates two concerns:

```
api/        ← "What does the web server expose?"   (HTTP routes, schemas, DB models)
src/        ← "How does detection actually work?"  (features, ML, graph, external APIs)
```

**Why?** This is the **separation of concerns** principle. If you want to swap FastAPI for
a different web framework tomorrow, you only touch `api/`. The core ML logic in `src/`
is completely unaware that it's inside a web server. This also makes `src/` independently
testable without needing to boot up the full HTTP server.

---

## The Singleton Pattern — `api/dependencies.py`

```python
# api/dependencies.py
_prediction_service = None

def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(...)
    return _prediction_service
```

**Why a singleton?** Loading two ML models (Random Forest + XGBoost), a graph file,
and a preprocessor takes several seconds and ~500MB of RAM. If the system created a
new `PredictionService` for every HTTP request, it would be unbearably slow.
The singleton ensures models are loaded **once** when the server starts and reused for
every subsequent request.

---

## The Feature Schema — `src/core/schema.py`

```python
# src/core/schema.py
FEATURE_ORDER = URL_FEATURES + DOMAIN_FEATURES + CONTENT_FEATURES + GRAPH_FEATURES
```

This is a **single source of truth** for what the 60 features are and what order they come in.

**Why does order matter?** When a Random Forest model is trained, it learns that "column 3
is domain_age_days". If you later feed it data where column 3 is something else, the prediction
is nonsense. `FEATURE_ORDER` ensures the live data always matches what the model was trained on.

```python
# api/services/prediction_service.py  (line 82-85)
if hasattr(self.ensemble_predictor.rf_model, 'feature_names_in_'):
    expected_cols = list(self.ensemble_predictor.rf_model.feature_names_in_)
    df_features = df_features.reindex(columns=expected_cols, fill_value=0)
```

`reindex` forces the DataFrame to only use columns the model expects, filling any missing
ones with 0. This prevents the "Feature names mismatch" error.

---

## Pydantic Schemas — `api/schemas/`

```python
# api/schemas/request.py
class PredictionRequest(BaseModel):
    url: str = Field(..., description="The URL to analyze")
    include_explanation: bool = Field(default=True)
    fetch_content: bool = Field(default=False)
```

**Why define separate request and response schemas?** It acts as a contract between the
frontend and backend. The `...` in `Field(...)` means the field is **required** — FastAPI
will return a 422 error automatically if someone forgets to include `url`.
Response schemas (like `PredictionResponse`) guarantee the frontend always gets predictable
JSON, even if internal structures change.

---

## Database Table Split — `api/models.py`

```python
class Prediction(Base):        # Stores: url, label, confidence, risk_score, timestamps
    ...
class FeatureVector(Base):     # Stores: the full 60-feature JSON blob
    ...
```

**Why two tables?** The `Prediction` table is small and fast to query — the analytics
dashboard can load 100 recent predictions without touching the heavy feature data.
The `FeatureVector` table stores the full 60 features as JSONB (PostgreSQL's binary JSON
format, which supports fast querying within the JSON). This is a **performance design pattern**
called "vertical partitioning" — split data by how often it's accessed.

---

## CORS Middleware — `api/main.py`

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow requests from any domain
    allow_methods=["*"],
    allow_headers=["*"]
)
```

**Why is this needed?** By default, browsers block JavaScript on one domain (e.g.
`localhost:8501` — Streamlit) from calling an API on another domain (`localhost:8000` — FastAPI).
CORS middleware tells the browser: "it's okay, this API accepts requests from anywhere."
In production, you would replace `["*"]` with a specific list of trusted domains.

---

## The Ensemble Architecture

```
Random Forest  ──┐
                 ├──[weighted average]──> final_score ──> "phishing" or "safe"
XGBoost        ──┤
                 │
TGIS Trust     ──┘
```

**Why three models?** Each catches different things:
- RF and XGBoost are strong on URL structure and domain metadata patterns.
- TGIS catches cases where a URL looks "clean" but its neighbors are all phishing sites.

The cold-start handling is a smart design detail:
```python
# src/models/ensemble.py  (line 95-104)
is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)
if is_cold_start:
    final_score = (0.5 * rf_proba) + (0.5 * xgb_proba)  # ignore TGIS
else:
    final_score = (0.4 * rf_proba) + (0.4 * xgb_proba) + (0.2 * (1 - tgis_score))
```

If a domain has never been seen before (not in the graph), TGIS defaults to 0.5 (neutral).
Including a neutral score would just dilute the ML models, so TGIS is excluded entirely
until the domain appears in the graph.

---

## The NaN Sanitization Layer

```python
# api/services/prediction_service.py  (line 104-106)
safe_base = self._sanitize_features(base_features)
safe_graph = self._sanitize_features(graph_features)
```

External lookups (WHOIS, DNS, SSL) can fail for many reasons.
When they do, features are stored as `np.nan` (Not a Number). JSON has no concept of NaN —
trying to serialize it crashes. The sanitizer converts all NaN values to `-1.0` before
building the HTTP response, a safe sentinel value that the frontend can recognize as "data unavailable."
