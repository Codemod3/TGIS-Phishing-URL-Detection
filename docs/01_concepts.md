# Core Concepts

This file explains the fundamental ideas behind the project before any code is shown.

---

## 1. What Is Phishing?

Phishing is a cyberattack where a malicious website pretends to be a trusted one
(e.g., a fake PayPal login page) to steal your password or credit card details.
This system tries to automatically detect whether a URL is phishing or safe.

---

## 2. Feature Engineering

The system does not "read" a URL the way a human does. Instead, it converts the URL
into a list of **numbers** (called a feature vector) that a machine learning model can process.

For example, the URL `http://paypa1-login.tk/steal.php?user=you` becomes:

```
url_length          = 44
num_hyphens         = 1
has_ip_address      = 0
tld_suspicious      = 1   (.tk is a known spam TLD)
domain_age_days     = 3   (brand new domain - suspicious)
domain_in_brand_list= 1   (contains "paypa" similar to PayPal)
...
```

The system extracts **60 such numbers** per URL, grouped into 4 categories:

| Group | Count | Examples |
|-------|-------|---------|
| URL Structure | 15 | length, number of dots, slashes, special characters |
| Domain Metadata | 20 | domain age, SSL validity, registrar reputation |
| Page Content | 15 | has login form, uses iframes, password fields |
| Graph Features | 10 | cluster risk, suspicious neighbors, trust score |

---

## 3. Machine Learning Models

### What is a Random Forest?
Imagine asking 100 different experts to each look at a URL and vote "phishing" or "safe".
The majority vote wins. That's a Random Forest — it's made of many decision trees, each
trained on slightly different data, and the final answer is their combined vote.

### What is XGBoost?
XGBoost (eXtreme Gradient Boosting) is another ML model. Unlike Random Forest where all
trees are independent, XGBoost builds trees **one after another**, where each new tree
learns from the mistakes of the previous one. It often outperforms Random Forest on structured data.

### What is an Ensemble?
An ensemble combines multiple models to get a better final answer.
This project uses a weighted combination:

```
final_score = (0.4 × RF probability) + (0.4 × XGBoost probability) + (0.2 × TGIS risk)
```

If one model is uncertain, the others compensate.

---

## 4. TGIS — Trust Graph Intelligence System

This is the most unique part of the project. Instead of looking at a URL in isolation,
TGIS asks: **"Who are this domain's neighbors, and are they trustworthy?"**

Think of it like a city map:
- Each domain, IP address, registrar, and nameserver is a **node** (a dot on the map).
- Connections between them are **edges** (roads on the map).
- A domain registered by a shady registrar, sharing an IP with known phishing sites,
  gets a **low trust score**.

The graph is a `networkx.DiGraph` — a directed graph (edges have a direction, like arrows).

```
google.com ──[DOMAIN_TO_REGISTRAR]──> MarkMonitor (trusted registrar)
evil-site.tk ──[DOMAIN_TO_IP]──> 185.220.x.x (shared with 50 phishing domains)
```

**Trust Propagation** is the algorithm that spreads trust scores through the graph:
- Known safe domains start at 1.0
- Known phishing domains start at 0.0
- Unknown domains start at 0.5 and drift toward their neighbors' scores over 10 iterations

---

## 5. REST API

A REST API is a way for two programs to talk to each other over the internet using HTTP
(the same protocol your browser uses). The frontend (Streamlit dashboard) sends a request
to the backend (FastAPI server) and gets a JSON response.

```
Dashboard ──POST /api/v1/predict──> FastAPI ──> ML Pipeline ──> Response JSON
```

**JSON** is a human-readable data format:
```json
{
  "prediction": "phishing",
  "confidence": 0.92,
  "risk_score": 0.87
}
```

---

## 6. Database & ORM

The system saves every prediction to a **PostgreSQL** database.
Instead of writing raw SQL like `INSERT INTO predictions VALUES (...)`,
it uses **SQLAlchemy ORM** — a library that lets you work with database rows as Python objects.

```python
new_prediction = Prediction(url="http://evil.tk", prediction_label="phishing")
db.add(new_prediction)
db.commit()  # This writes to the database
```

---

## 7. Pydantic — Data Validation

**Pydantic** is a Python library that defines exactly what shape data must have.
If someone sends `{"url": 12345}` (a number instead of a string), Pydantic rejects it
automatically before it even reaches your code. It is used to define both request
(incoming data) and response (outgoing data) shapes.

---

## 8. WHOIS

WHOIS is an internet protocol that tells you who registered a domain and when.
A 3-day-old domain is far more suspicious than a 10-year-old one.
The system uses the `python-whois` library to query this information.
