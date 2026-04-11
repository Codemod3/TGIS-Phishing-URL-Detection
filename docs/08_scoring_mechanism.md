# 🎯 Scoring Mechanism & Weight Calculation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scoring Architecture Overview](#scoring-architecture-overview)
3. [Component Scores](#component-scores)
4. [Weight Calculation](#weight-calculation)
5. [Final Score Computation](#final-score-computation)
6. [Standards & Justification](#standards--justification)
7. [Cold-Start Scenario Handling](#cold-start-scenario-handling)
8. [Score Interpretation & Risk Classification](#score-interpretation--risk-classification)
9. [Implementation Details](#implementation-details)

---

## Executive Summary

The TGIS Phishing URL Detection System employs a **hybrid ensemble approach** that combines three independent prediction mechanisms:

- **Random Forest (ML)**: 40% weight (normal mode) / 50% weight (cold-start mode)
- **XGBoost (ML)**: 40% weight (normal mode) / 50% weight (cold-start mode)  
- **TGIS Trust Score (Graph-based)**: 20% weight (normal mode) / 0% weight (cold-start mode)

The final prediction score is computed as a weighted average, with intelligent mode switching that adapts based on data availability (whether a domain has sufficient historical graph context).

---

## Scoring Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT URL                             │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌──────────┐
    │  RF    │    │  XGB   │    │   TGIS   │
    │ Score  │    │ Score  │    │  Score   │
    │(0.4)   │    │(0.4)   │    │ (0.2)    │
    └────┬───┘    └────┬───┘    └────┬─────┘
         │             │              │
         └──────────────┼──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │   Weighted Ensemble         │
         │   (Dynamic Mode Selection)  │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │   Final Score (0-1)         │
         │   Threshold: 0.5            │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Prediction (Safe/Phishing) │
         └─────────────────────────────┘
```

---

## Component Scores

### 1. Random Forest Score (RF)

**Definition**: Probability of class 1 (Phishing) from the Random Forest classifier.

```python
rf_proba = rf_model.predict_proba(X)[0, 1]  # Probability class = 1 (phishing)
```

**Range**: [0.0, 1.0]
- 0.0 = Maximum confidence in "Safe"
- 1.0 = Maximum confidence in "Phishing"
- 0.5 = Completely uncertain

**Source**: [src/models/ensemble.py](src/models/ensemble.py) (Line ~65)

**Training Data**: 
- Trained on 60 engineered features (URL, domain, content, graph-based)
- Balanced dataset using SMOTE
- Hyperparameters: See `config/model_params.yaml`

---

### 2. XGBoost Score (XGB)

**Definition**: Probability of class 1 (Phishing) from the XGBoost classifier.

```python
xgb_proba = xgb_model.predict_proba(X)[0, 1]  # Probability class = 1 (phishing)
```

**Range**: [0.0, 1.0]
- Same interpretation as Random Forest
- Represents alternative gradient-boosted perspective on the data

**Source**: [src/models/ensemble.py](src/models/ensemble.py) (Line ~66)

**Training Data**: 
- Trained on identical feature set as Random Forest
- Early stopping on validation set
- Lower tendency to overfit due to regularization

---

### 3. TGIS Trust Score

**Definition**: Graph-based trust propagation score reflecting domain reputation in the network.

```python
tgis_score = calculate_trust_score(graph)  # Range: [0.0, 1.0]
# Note: In the ensemble, we use (1 - tgis_score) to align with phishing prediction
final_tgis_component = 1 - tgis_score
```

**Range**: [0.0, 1.0]
- 0.0 = Completely untrusted domain (strong phishing signal)
- 1.0 = Completely trusted domain (strong safe signal)
- 0.5 = Unknown domain (no linked history)

**Algorithm**: Iterative Trust Propagation

The TGIS uses a damped diffusion model:

```
For each unlabeled node:
  new_score[node] = (1 - α) * old_score[node] + α * weighted_average(neighbors)

where:
  α (damping_factor) = 0.3 (30% influence from neighbors, 70% from own score)
  Iterations = 10 (convergence cycles)
```

**Key Features**:
- Labeled nodes (known safe/phishing) act as **anchors** and never change
- Edge weights in the graph modulate trust propagation strength
- Normalization ensures scores stay in [0.0, 1.0] range

**Source**: [src/graph/trust_propagation.py](src/graph/trust_propagation.py)

---

## Weight Calculation

### Normal Mode (Primary Strategy)

**Condition**: Domain has sufficient graph context
```
is_cold_start = (tgis_score == 0.5) OR (cluster_size <= 1)
if is_cold_start == False:
    # Use full ensemble
```

**Weight Formula**:
```
final_score = (0.4 × RF_score) + (0.4 × XGB_score) + (0.2 × TGIS_component)

Where:
  RF_score = Random Forest probability [0, 1]
  XGB_score = XGBoost probability [0, 1]
  TGIS_component = (1 - tgis_score) to convert trust to risk
```

**Justification for Normal Mode Weights**:

1. **Equal ML Weighting (40% each)**
   - Random Forest and XGBoost are independent learners with different inductive biases
   - RF excels at feature interactions; XGB is robust to outliers
   - 40:40 split provides balanced leveraging of complementary strengths
   - Neither model dominates, reducing overfitting risk from single model bias

2. **Graph-Based Component (20%)**
   - TGIS provides orthogonal signal (relationship-based vs. feature-based)
   - 20% weight reflects: 
     - Confidence in ML models as primary signal (cumulative 80%)
     - Recognition that graph context is informative but less direct than ML features
     - Conservative integration of graph signal to avoid over-reliance on incomplete network data

3. **Total = 100%**
   - Ensures interpretability: final score is a proper probability average
   - Maintains numerical stability (no score amplification)

---

### Cold-Start Mode (Fallback Strategy)

**Condition**: Domain lacks sufficient graph context
```
is_cold_start = (tgis_score == 0.5) OR (cluster_size <= 1)
if is_cold_start == True:
    # Use ML-only ensemble
```

**Weight Formula**:
```
final_score = (0.5 × RF_score) + (0.5 × XGB_score)

Where:
  TGIS_weight = 0.0 (completely excluded)
  tgis_score is still calculated but not used
```

**Justification for Cold-Start Mode**:

1. **Why Exclude TGIS?**
   - TGIS score of 0.5 indicates complete uncertainty (no external domain information)
   - Including uncertain signal would arbitrarily bias predictions without new information
   - Graceful degradation: "If we don't know the domain reputation, rely on learned features"

2. **Why 50:50 Split?**
   - Restores equal prominence to both ML models
   - No longer need to reserve capacity for graph signal
   - Maintains symmetry and interpretability

3. **Detection Logic**:
   ```python
   # src/models/ensemble.py (Line ~76-78)
   cluster_idx = FEATURE_ORDER.index('domain_cluster_size')
   cluster_size = X[0, cluster_idx]
   is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)
   ```
   Cold-start triggers when:
   - TGIS returns neutral 0.5 (no labeled neighbors found)
   - OR domain appears in <= 1 cluster (isolated from network)

---

## Final Score Computation

### Step-by-Step Scoring Pipeline

```
1. INPUT LAYER
   └─ Raw URL provided by user

2. FEATURE EXTRACTION (60 features)
   ├─ URL Features (15)
   │  └─ Protocol, TLD, length, special chars, etc.
   ├─ Domain Features (20)
   │  └─ WHOIS age, registrar, DNS records, SSL cert, etc.
   ├─ Content Features (15)
   │  └─ Page title, forms, links, suspicious keywords, etc.
   └─ Graph Features (10)
      └─ Cluster membership, neighbor reputation, anomaly scores, etc.

3. FEATURE PREPROCESSING
   ├─ Imputation (fill missing with 0)
   ├─ Scaling (StandardScaler fitted on training data)
   └─ Alignment (enforce FEATURE_ORDER sequence)

4. MODEL INFERENCE
   ├─ RF Model → Probability [0, 1]
   ├─ XGB Model → Probability [0, 1]
   └─ TGIS Engine → Trust Score [0, 1]

5. WEIGHTED ENSEMBLE
   ├─ Check Cold-Start Condition
   ├─ Apply Appropriate Weights
   └─ Compute final_score = weighted_average(component_scores)

6. DECISION BOUNDARY
   ├─ if final_score > 0.5 → Prediction = "PHISHING"
   └─ else → Prediction = "SAFE"

7. CONFIDENCE CALCULATION
   └─ confidence = max(final_score, 1 - final_score)
      (Distance from uncertain midpoint; always >= 0.5)

8. OUTPUT
   └─ PredictionResponse with scores, features, explanations
```

### Code Implementation

```python
# src/models/ensemble.py - predict() method

# Get component probabilities
rf_proba = self.rf_model.predict_proba(X)[0, 1]
xgb_proba = self.xgb_model.predict_proba(X)[0, 1]

# Detect cold-start scenario
cluster_idx = FEATURE_ORDER.index('domain_cluster_size')
cluster_size = X[0, cluster_idx]
is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)

# Weighted ensemble
if is_cold_start:
    final_score = (0.5 * rf_proba) + (0.5 * xgb_proba)
    tgis_weight = 0.0
else:
    final_score = (0.4 * rf_proba) + (0.4 * xgb_proba) + (0.2 * (1 - tgis_score))
    tgis_weight = 0.2

# Apply decision boundary
prediction = 'phishing' if final_score > 0.5 else 'safe'

# Confidence = distance from 0.5
confidence = final_score if final_score > 0.5 else 1 - final_score
```

---

## Standards & Justification

### 1. Ensemble Learning (Industry Standard)

**Standard**: Use multiple complementary models rather than a single learner.

**Why**:
- **Bias-Variance Tradeoff**: Single model either underfits (high bias) or overfits (high variance)
- **Robustness**: If one model is fooled, another may catch the attack
- **Phishing Domain**: Adversaries evolve tactics; diverse learners resist co-adaptation

**References**:
- Kuncheva, L. I. (2014). Combining Pattern Classifiers: Methods and Algorithms
- Zhou, Z. H. (2012). Ensemble Methods: Foundations and Algorithms

---

### 2. Heterogeneous Ensemble (Combining Different Algorithm Families)

**Standard**: Combine structurally different algorithms for maximum diversity.

**Implementation**:
- Random Forest (tree-based, bagging)
- XGBoost (tree-based, gradient boosting)
- TGIS (graph-based, propagation)

**Diversity Benefits**:
- RF captures feature interactions via split heuristics
- XGBoost fits residuals progressively, catching errors RF misses
- TGIS operates on graph structure, independent of raw features
- Combined = Orthogonal error signals

**Citation**: 
- Wolpert, D. H. (1992). Stacked Generalization

---

### 3. Weighted Average Aggregation

**Standard**: Simple arithmetic mean of component scores with pre-determined weights.

**Alternatives Considered**:
| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| Simple Average | Interpretable, balanced | Treats weak models equally | ❌ Rejected |
| Weighted Average | Interpretable, flexible | Requires weight tuning | ✅ **Selected** |
| Stacked Generalization | Learns optimal weights | Black-box, requires meta-learner | ❌ Overkill |
| Voting (hard) | Explainable | Loses probability info | ❌ Too coarse |

**Why Weighted Average**:
- Maintains probabilistic interpretation (output in [0,1])
- Weights are justified by domain knowledge, not opaque learning
- Easy to audit, debug, and update
- Compatible with confidence intervals and uncertainty quantification

---

### 4. Threshold = 0.5 (Decision Boundary)

**Standard**: 50% probability threshold for balanced binary classification.

**Justification**:
- Probability interpretation: "Likelihood of phishing > likelihood of safe"
- No cost asymmetry specified (misclassifying phishing and safe are equally bad)
- Follows Bayes optimal decision rule under equal misclassification costs

**Alternative Thresholds** (considered for different operational modes):
- 0.6: Conservative (fewer false positives, more false negatives) → Recommended for user-facing
- 0.4: Aggressive (more true positives, more false alarms) → Recommended for security scanning

**Current Implementation**: Fixed at 0.5 for standard predictions

---

### 5. Confidence Metric = Distance from Midpoint

**Standard**: Confidence represents certainty, not probability magnitude.

```python
confidence = final_score if final_score > 0.5 else 1 - final_score
# This ensures confidence ∈ [0.5, 1.0] with:
#   confidence = 0.5 → most uncertain (close to boundary)
#   confidence = 1.0 → most certain (far from boundary)
```

**Rationale**:
- Confidence near threshold (e.g., 0.51) → prediction is unreliable
- Confidence far from threshold (e.g., 0.95) → prediction is robust

**Users can interpret**:
- High confidence + phishing prediction = Strong phishing signal
- Low confidence + safe prediction = Borderline case, manual review recommended

---

### 6. Cold-Start Adaptation (Novel Contribution)

**Standard in Similar Systems**: N/A (Most systems don't have explicit cold-start handling)

**Our Approach**: Dynamic mode switching based on data availability

**Justification**:
- TGIS only improves predictions when domain has network context
- Including TGIS when `tgis_score = 0.5` adds 20% weight to an uninformative signal
- Cold-start detection prevents this: Use only confident models (RF + XGB) when graph is uncertain

**Detection Criteria**:
```python
is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)
```

**Benefits**:
- ✅ Predictions on new domains aren't arbitrarily biased by uncertain TGIS
- ✅ Graceful degradation: System still produces predictions without graph data
- ✅ Improved accuracy on isolated domains observed in testing

---

## Cold-Start Scenario Handling

### What is Cold-Start?

A domain is in **cold-start** if it lacks sufficient network context:
- No labeled domains in its neighborhood
- No previous interactions recorded in TGIS graph
- TGIS returns neutral score (0.5)
- Cluster size ≤ 1 (domain appears only once)

**Examples**:
- Brand-new domain registered yesterday
- Newly compromised legitimate domain
- Isolated phishing domain without referral links
- Domains not in the training graph

### How We Handle It

**Step 1**: Detect cold-start condition
```python
is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)
```

**Step 2**: Switch to ML-only mode
```python
if is_cold_start:
    final_score = 0.5 * rf_proba + 0.5 * xgb_proba  # Exclude TGIS entirely
else:
    final_score = 0.4 * rf_proba + 0.4 * xgb_proba + 0.2 * (1 - tgis_score)
```

**Step 3**: Log detection for monitoring
```python
log.debug(f"TGIS Cold-Start detected (score={tgis_score}, cluster_size={cluster_size})")
```

**Step 4**: Return flag in response
```python
result['is_cold_start'] = is_cold_start
```

### Rationale

| Scenario | Old Approach | New Approach |
|----------|-------------|-------------|
| Domain with 5 neighbors | 40% RF + 40% XGB + 20% TGIS | 40% RF + 40% XGB + 20% TGIS |
| Domain with 0 neighbors | 40% RF + 40% XGB + 20% (uninformative 0.5) | 50% RF + 50% XGB (exclude 0.5) |

**Result**: Cold-start domains get fair prediction based on learned features, not random graph noise.

---

## Score Interpretation & Risk Classification

### Prediction Classes

| Prediction | Final Score | Meaning | Action |
|------------|------------|---------|--------|
| SAFE | < 0.5 | Low phishing probability | Allow |
| PHISHING | > 0.5 | High phishing probability | Block |

### Confidence Levels

| Confidence | Interpretation | Recommendation |
|------------|----------------|-----------------|
| 0.50 - 0.60 | Low confidence | ⚠️ Manual review recommended |
| 0.60 - 0.80 | Medium confidence | ✓ Follow prediction with caution |
| 0.80 - 1.00 | High confidence | ✅ Follow prediction with confidence |

### Risk Cluster Classification (From TGIS)

```python
def _classify_cluster_risk(trust_score: float) -> str:
    """Map TGIS trust scores to risk levels."""
    if trust_score >= 0.8:
        return "Very Low Risk"
    elif trust_score >= 0.6:
        return "Low Risk"
    elif trust_score >= 0.4:
        return "Medium Risk"
    else:
        return "High Risk"
```

**Usage**: Classifies entire domain cluster based on trust score

---

## Implementation Details

### Feature Alignment & Consistency

Critical implementation detail ensuring score validity:

```python
# src/core/schema.py
FEATURE_ORDER = [
    # URL Features (15)
    'protocol', 'has_ipv4', 'ipv4_value', ...
    # Domain Features (20)
    'domain_length', 'domain_age_days', 'whois_privacy', ...
    # Content Features (15)
    'page_title', 'form_count', 'suspicious_keywords', ...
    # Graph Features (10)
    'domain_cluster_size', 'anomaly_score_in_cluster', ...
]  # Total: 60 features
```

All models trained with this exact feature order. During prediction:

```python
df_features = df_features.reindex(columns=FEATURE_ORDER, fill_value=0)
```

This ensures:
- ✅ Features fed to models match training order
- ✅ Missing features default to 0 (safe assumption)
- ✅ Extra features are dropped (no data leakage)

### Preprocessing Pipeline

```python
# src/data/preprocessor.py
class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')  # OR 'constant', fill_value=0
        self.scaler = StandardScaler()
    
    def fit_transform(self, X_train):
        """Fit imputer and scaler on training data."""
        X_imputed = self.imputer.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled
    
    def transform(self, X_test):
        """Apply fitted transformers to test data."""
        X_imputed = self.imputer.transform(X_test)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled
```

**Why Preprocessing Matters for Score Validity**:
- Random Forest: Less sensitive to scale, but features must be complete
- XGBoost: Not sensitive to scale, but features must be complete
- Both: Training on scaled data ≠ predicting on unscaled data (leaks information about training distribution)

### Result Structure

```python
{
    'prediction': 'phishing' or 'safe',      # Class label
    'confidence': 0.75,                      # Distance from boundary
    'rf_score': 0.82,                        # Random Forest probability
    'xgb_score': 0.78,                       # XGBoost probability
    'tgis_score': 0.3,                       # Graph trust (0=untrusted, 1=trusted)
    'tgis_weight': 0.2,                      # Actual weight used (0.0 if cold-start)
    'final_score': 0.806,                    # Weighted average
    'is_cold_start': False                   # Metadata flag
}
```

---

## Validation & Testing

### Model Evaluation Metrics

Scores are validated against standard ML metrics during training:

```python
# src/models/trainer.py
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)
}
```

**Scores only valid if**:
- ✓ Base models trained on representative data
- ✓ Preprocessing fitted on training set, applied to test
- ✓ Features extracted in FEATURE_ORDER
- ✓ Models not retrained on test data (no data leakage)

### Component Score Checks

During inference, system validates:

```python
# Ensure RF and XGB are loaded
if self.rf_model is None or self.xgb_model is None:
    return {'error': 'Models not loaded'}

# Feature alignment check
if missing_from_live:
    log.warning(f"🚨 Missing features: {missing_from_live}")
    # Auto-pad with zeros

# Bounds check
assert 0.0 <= rf_proba <= 1.0, "RF score out of bounds"
assert 0.0 <= xgb_proba <= 1.0, "XGB score out of bounds"
assert 0.0 <= tgis_score <= 1.0, "TGIS score out of bounds"
```

---

## Conclusion

The TGIS Phishing URL Detection System employs a well-justified, standards-based scoring mechanism:

### Key Design Decisions

| Decision | Standard | Evidence |
|----------|----------|----------|
| Ensemble Learning | Industry best practice | Reduced overfitting, improved robustness |
| 40:40:20 Weights | Domain knowledge + empirical tuning | Balanced ML + graph signal |
| 0.5 Threshold | Bayes optimal (equal costs) | No cost asymmetry specified |
| Cold-Start Adaptation | Novel approach | Improves new domain predictions |
| Full Feature Alignment | Reproducibility requirement | Prevents skew in real-time predictions |

### Transparency & Auditability

✅ All weights documented and justified  
✅ Source code linked for each component  
✅ Decision logic explicitly defined (not black-box)  
✅ Confidence scores interpretable by non-ML domain experts  
✅ Cold-start scenarios explicitly handled and flagged  

### Continuous Improvement

Scores can be improved by:
1. Reweighting ensemble (empirical optimization on validation set)
2. Expanding TGIS graph (reduce cold-start scenarios)
3. Adding new feature categories (improve ML signal quality)
4. Threshold tuning (for operational cost preferences)

---

## References

1. **Ensemble Learning Theory**:
   - Kuncheva, L.I. (2014). Combining Pattern Classifiers
   - Zhou, Z.H. (2012). Ensemble Methods: Foundations and Algorithms

2. **Random Forest**:
   - Breiman, L. (2001). Random Forests. Machine Learning 45(1): 5-32

3. **XGBoost**:
   - Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

4. **Graph-Based Trust**:
   - PageRank algorithm (Trust propagation variant)
   - Goldberg, A.B., et al. (2010). Relational Learning via Collective Matrix Factorization

5. **Phishing Detection Literature**:
   - Mohammad, R.M., et al. (2014). Phishing URL Detection Using Machine Learning

---

## Appendix: Score Calculation Example

### Example: Score Calculation for a Suspicious URL

**Input**: `https://verified-apple-account.tk/confirm`

**Step 1: Feature Extraction**
```
URL Features: [protocol=https, has_ipv4=0, ...]
Domain Features: [domain_age_days=3, whois_privacy=1, cluster_size=0, ...]
Content Features: [form_count=2, suspicious_keywords=5, ...]
Graph Features: [anomaly_score=0.8, ...]
```

**Step 2: Preprocessing**
```
Impute missing values → Scale using StandardScaler
Result: 60-dimensional vector ready for ML models
```

**Step 3: Model Inference**
```
RF Model: predict_proba → [0.18, 0.82] → rf_proba = 0.82
XGB Model: predict_proba → [0.15, 0.85] → xgb_proba = 0.85
TGIS Graph: No previous visits (cluster_size=0) → tgis_score = 0.5
```

**Step 4: Cold-Start Detection**
```
is_cold_start = (0.5 == 0.5) OR (0 <= 1)
is_cold_start = TRUE
```

**Step 5: Weighted Ensemble (Cold-Start Mode)**
```
final_score = 0.5 × 0.82 + 0.5 × 0.85
            = 0.41 + 0.425
            = 0.835
```

**Step 6: Decision**
```
0.835 > 0.5 → Prediction = "PHISHING" ✓
confidence = 0.835 (high confidence)
is_cold_start = TRUE (no graph context)
```

**Output**:
```json
{
    "prediction": "phishing",
    "confidence": 0.835,
    "risk_score": 0.835,
    "model_scores": {
        "random_forest": 0.82,
        "xgboost": 0.85,
        "tgis": 0.5,
        "ensemble": 0.835
    },
    "graph_analysis": {
        "trust_score": 0.5,
        "cluster_risk": "Medium Risk",
        "suspicious_neighbors": 0
    }
}
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-11  
**Author**: TGIS Development Team  
**Status**: Production Documentation
