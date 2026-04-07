# Middleware, API Layer & Frontend

This file covers how the web server is wired up, how routes work, and how the Streamlit
dashboard connects to and displays data from the backend.

---

## 1. FastAPI Application Entry Point — `api/main.py`

```python
# api/main.py  (line 6-10)
app = FastAPI(
    title="Phishing URL Detector API",
    version="1.0.0",
    description="Elite phishing detection combining ML and Trust Graph Intelligence (TGIS)"
)
```

This creates the FastAPI application. FastAPI automatically generates interactive API
documentation from this metadata, available at `http://localhost:8000/docs`.
You can test every endpoint directly in your browser — no extra tools needed.

```python
# api/main.py  (line 12-22)
@app.on_event("startup")
def setup_database():
    try:
        from api.database import engine, Base
        import api.models
        Base.metadata.create_all(bind=engine)  # Creates tables if they don't exist
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        print("System will continue without persistence layer.")
```

`@app.on_event("startup")` is a decorator that registers a function to run once when
the server starts. `create_all` checks if the tables exist and creates them if not —
it does **not** drop existing data. The try/except means the server boots even if the
database is down — it just won't save results.

---

## 2. CORS Middleware — `api/main.py`

```python
# api/main.py  (line 25-31)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

**Middleware** is code that runs on every request and response, like a checkpoint.
CORS (Cross-Origin Resource Sharing) is a browser security rule. The Streamlit app runs
on port 8501, but the API is on port 8000 — different ports = different "origins".
Browsers block such requests by default. `allow_origins=["*"]` disables that check,
allowing the dashboard to freely call the API. In production, replace `"*"` with the
specific URL of your dashboard.

---

## 3. Routers — `api/routes/`

Instead of putting all routes in `main.py`, the app uses **APIRouter** to group
related endpoints into separate files.

```python
# api/routes/predict.py  (line 10-33)
router = APIRouter(prefix="/api/v1", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict_url(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
    db: Session = Depends(get_db)
):
    return service.predict_single_url(request.url, db=db)
```

- `prefix="/api/v1"` means this route is reached at `/api/v1/predict`.
- `@router.post` means it accepts HTTP POST requests (sending data to the server).
- `response_model=PredictionResponse` tells FastAPI to validate and serialize the return
  value using the Pydantic schema. If a field is missing or the wrong type, FastAPI catches it.
- `Depends(get_prediction_service)` is **Dependency Injection** — FastAPI calls
  `get_prediction_service()` automatically and passes the result as `service`.
  This is how the singleton pattern connects to the route.
- `Depends(get_db)` similarly injects a fresh database session per request.

```python
# api/routes/history.py  (line 10-30)
@router.get("/history", response_model=HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    predictions = db.query(Prediction)\
        .order_by(Prediction.created_at.desc())\
        .limit(100).all()
    return HistoryResponse(total=len(predictions), results=predictions)
```

`db.query(Prediction)` is SQLAlchemy ORM syntax. It builds a SQL query like:
`SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100`.
The result is a list of `Prediction` Python objects — no raw SQL needed.

```python
# api/routes/health.py  (line 13-44)
START_TIME = time.time()  # Set once when the module is loaded

@router.get("/health", response_model=HealthResponse)
def health_check():
    uptime = int(time.time() - START_TIME)  # Seconds since server started
    return HealthResponse(status="healthy", uptime_seconds=uptime, ...)
```

`time.time()` returns the current time as a Unix timestamp (seconds since Jan 1, 1970).
Subtracting `START_TIME` gives uptime. This is a standard monitoring endpoint — load
balancers and monitoring systems ping `/health` to check if the service is alive.

---

## 4. Pydantic Schemas in Detail — `api/schemas/`

```python
# api/schemas/request.py  (line 4-8)
class PredictionRequest(BaseModel):
    url: str = Field(..., description="The URL to analyze")
    include_explanation: bool = Field(default=True)
    fetch_content: bool = Field(default=False)
```

`BaseModel` is the parent class from Pydantic. Every class that inherits from it gets:
- Automatic type validation (wrong type = immediate 422 error response)
- `.json()` method to convert to JSON
- Auto-generated documentation in FastAPI's `/docs` page

`Field(...)` — the `...` (called Ellipsis) means the field is required.
`Field(default=True)` means it's optional and defaults to `True` if not provided.

```python
# api/schemas/response.py  (line 100-101)
class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True, ...)
```

`from_attributes=True` tells Pydantic it can read data from object attributes, not
just dictionaries. This is needed because SQLAlchemy ORM returns objects (e.g. `prediction.url`),
not plain dictionaries. Without this, Pydantic couldn't convert them to JSON.

---

## 5. WHOIS Client with Caching — `src/external/whois_client.py`

```python
# src/external/whois_client.py  (line 28-48)
def lookup(self, domain: str):
    cache_key = f"whois:{domain}"
    cached_data = cache_manager.get(cache_key)  # Check Redis first
    if cached_data:
        return cached_data                        # Return instantly if cached

    w = whois.whois(domain)           # Real network call (slow ~1-3 seconds)
    result = self._parse_whois(w)
    cache_manager.set(cache_key, result, ttl=604800)  # Cache for 7 days
    return result
```

WHOIS lookups are slow (network calls to registrar servers). Caching the result in Redis
for 7 days means the second request for `google.com` returns in milliseconds instead of seconds.
`ttl=604800` is 7 days in seconds (7 × 24 × 60 × 60 = 604800).

---

## 6. Streamlit Dashboard Entry Point — `dashboard/app.py`

```python
# dashboard/app.py  (line 13-18)
st.set_page_config(
    page_title="Phishing URL Detector | Elite Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

`st.set_page_config` must be the first Streamlit call. It configures the browser tab title,
icon, and layout. `layout="wide"` uses the full browser width instead of a narrow centered column.

```python
# dashboard/app.py  (line 65-88)
with st.sidebar:
    health = client.get_health()
    is_online = health["status"] == "healthy"
    status_display = "🟢 ONLINE" if is_online else "🔴 OFFLINE"
    st.markdown(f"**Backend Status:** {status_display}")

    menu = st.radio("Select Workspace",
        options=["🔍 Real-time Analysis", "📊 Batch Processing", "📈 Performance Logs"])
```

Streamlit re-runs the **entire script** every time a user interacts with any widget.
`st.radio` creates a radio button group. Its current selection is returned as a string,
which drives the `if/elif` routing below to show different page content.
The sidebar health check calls the `/health` API endpoint on every page reload.

---

## 7. API Client — `dashboard/utils/api_client.py`

```python
# dashboard/utils/api_client.py  (line 10-27)
def predict_url(self, url: str) -> Dict[str, Any]:
    payload = {"url": url, "include_explanation": True, "fetch_content": True}
    response = requests.post(
        f"{self.base_url}/api/v1/predict",
        json=payload,
        timeout=60
    )
    if response.status_code == 200:
        return response.json()       # Parse the JSON response body
    return {"error": f"API Error ({response.status_code}): {response.text}"}
```

`requests` is Python's most popular HTTP library. `requests.post(url, json=payload)` sends
a POST request with the payload serialized as JSON. `response.json()` parses the JSON
response body back into a Python dictionary. The `timeout=60` prevents the UI from
freezing forever if the backend hangs.

---

## 8. Predictor UI Component — `dashboard/components/predictor.py`

```python
# dashboard/components/predictor.py  (line 44-50)
verdict = result["prediction"].upper()
color = "#ff4b4b" if verdict == "PHISHING" else "#00c853"

st.markdown(f"""
    <div style='background-color: {color}22; ...'>
        <h4 style='color: {color};'>VERDICT: {verdict}</h3>
    </div>
""", unsafe_allow_html=True)
```

`unsafe_allow_html=True` lets you inject raw HTML into Streamlit pages.
The `{color}22` suffix adds `22` to the hex color code — that's the alpha (transparency)
channel in 8-digit hex color notation. This creates a colored box with a semi-transparent background.

```python
# dashboard/components/predictor.py  (line 72-80)
chart_data = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "TGIS Graph", "Ensemble Summary"],
    "Phishing Probability": [
        scores["random_forest"],
        scores["xgboost"],
        1 - scores["tgis"],    # Invert TGIS: high trust = low phishing
        scores["ensemble"]
    ]
})
st.bar_chart(chart_data, x="Model", y="Phishing Probability")
```

`pd.DataFrame` creates a table from a dictionary of column_name → list_of_values.
`st.bar_chart` renders it as an interactive bar chart with zero configuration needed.

---

## 9. Analytics Tab — `dashboard/components/analytics.py`

```python
# dashboard/components/analytics.py  (line 56-71)
fig_ratio = px.pie(
    df,
    names='prediction_label',
    hole=0.6,                        # 0.6 makes it a donut chart
    color_discrete_map={'phishing': '#FF4B4B', 'safe': '#00C781'}
)
st.plotly_chart(fig_ratio, use_container_width=True)
```

`plotly.express` (imported as `px`) creates interactive charts. Unlike `st.bar_chart`
(Streamlit's built-in), Plotly charts are interactive — users can hover, zoom, and click.
`hole=0.6` turns a pie chart into a donut chart. `use_container_width=True` makes the
chart fill its column width responsively.

```python
# dashboard/components/analytics.py  (line 101-112)
st.dataframe(
    display_df,
    column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
        "URL": st.column_config.LinkColumn("Analyzed URL"),
    }
)
```

`st.dataframe` with `column_config` renders a rich, styled data table.
`ProgressColumn` shows a confidence bar (e.g., 87% filled) instead of a raw number.
`LinkColumn` makes URLs clickable. This is Streamlit's built-in alternative to writing
custom HTML tables.
