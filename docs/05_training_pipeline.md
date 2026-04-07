# ML Training Pipeline

This file explains how the Random Forest and XGBoost model files (`.pkl`) were produced.
The trained models are stored in `data/models/` and loaded at server startup.
Understanding training is important because it explains *why* the models behave the way they do.

---

## 1. The Abstract Base Class — `src/features/base.py`

Before any feature extractor runs, they all share a common contract:

```python
# src/features/base.py  (line 4-21)
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, url: str, **kwargs) -> Dict[str, Any]:
        pass
```

`ABC` stands for Abstract Base Class. Think of it as a template or a rule:
- Any class that inherits from `FeatureExtractor` **must** implement an `extract()` method.
- If it doesn't, Python will raise an error the moment you try to create an instance of it.
- This guarantees that `URLFeatureExtractor`, `DomainFeatureExtractor`,
  `ContentFeatureExtractor`, and `GraphFeatureExtractor` all have identical method signatures,
  making them interchangeable and reliably callable by `FeaturePipeline`.

`@abstractmethod` is a decorator that marks the method as "must be overridden".
`pass` means the base class has no implementation — it only defines the shape.

---

## 2. Data Splitting — `src/data/splitter.py`

Before training, the dataset is divided into three parts:

```python
# src/data/splitter.py  (line 32-43)
# Step 1: carve out the test set (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Step 2: carve out validation from what's left (15% of total)
val_size_adj = 0.15 / (1.0 - 0.15)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size_adj, random_state=42, stratify=y_train_val
)
```

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% | The model learns patterns from this data |
| Validation | 15% | Used during XGBoost training to detect overfitting (early stopping) |
| Test | 15% | Final evaluation — the model has never seen this data |

**Why three splits instead of two?**
If you tune the model on the test set, you're effectively "cheating" — you'd report
inflated performance. The validation set lets you tune without contaminating the test evaluation.

`stratify=y` ensures that the same ratio of phishing/safe URLs appears in each split.
Without it, one split could accidentally end up with 90% phishing and give misleading results.

`random_state=42` is just a fixed seed for the random number generator — it makes splits
reproducible. Running training twice gives identical splits.

---

## 3. SMOTE — Handling Imbalanced Data — `src/data/splitter.py`

```python
# src/data/splitter.py  (line 46-84)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**The problem:** Real-world datasets often have far fewer phishing URLs than safe ones.
If 90% of training data is "safe", a model that always predicts "safe" is 90% accurate
but completely useless — it never catches phishing.

**What SMOTE does (Synthetic Minority Over-sampling Technique):**
SMOTE doesn't just copy minority-class samples. Instead, it **generates new synthetic ones**
by interpolating between existing phishing samples. For each phishing URL, it finds its
`k` nearest neighbors and creates new fake-but-plausible examples along the line between them.
The result is a balanced dataset where both classes have equal representation.

**Why only on training data?**
```python
X_train_balanced, y_train_balanced = self.splitter.apply_smote(X_train_sub, y_train_sub)
# SMOTE is NOT applied to X_val or X_test
```
Validation and test sets must reflect real-world distribution. Generating synthetic test
data would produce fake metrics — you'd be measuring performance on made-up examples.

---

## 4. Random Forest Configuration — `src/models/random_forest.py`

```python
# src/models/random_forest.py  (line 7-17)
RF_PARAMS = {
    'n_estimators': 500,       # Build 500 decision trees
    'max_depth': 20,           # Each tree can be at most 20 levels deep
    'min_samples_split': 10,   # A node needs 10+ samples before it can split
    'min_samples_leaf': 5,     # Every leaf must contain at least 5 samples
    'max_features': 'sqrt',    # Each tree only sees sqrt(60) ≈ 8 random features
    'class_weight': 'balanced',# Automatically weights phishing class higher
    'random_state': 42,
    'n_jobs': -1,              # Use all available CPU cores
    'oob_score': True          # Calculate Out-of-Bag accuracy during training
}
```

**Key parameters explained:**
- `n_estimators=500`: More trees = more stable predictions, but slower training. 500 is
  a good balance for 60 features.
- `max_features='sqrt'`: Each tree only sees a random subset of features. This forces
  diversity among the 500 trees — if one misleading feature exists, not all trees see it.
- `class_weight='balanced'`: Tells the model: "treat each phishing sample as if it were
  heavier than a safe sample, proportional to class imbalance." A form of SMOTE via weights.
- `oob_score=True`: Out-of-Bag scoring. When building each tree, ~37% of training samples
  are left out (not sampled). The model evaluates itself on those "out of bag" samples,
  giving a free cross-validation estimate without needing a separate validation set.

---

## 5. XGBoost Configuration — `src/models/xgboost_model.py`

```python
# src/models/xgboost_model.py  (line 7-21)
XGB_PARAMS = {
    'n_estimators': 300,       # Build up to 300 trees (can stop early)
    'max_depth': 8,            # Shallower than RF — XGBoost prefers this
    'learning_rate': 0.1,      # How much each new tree corrects the previous error
    'subsample': 0.8,          # Each tree trains on 80% of the data (random)
    'colsample_bytree': 0.8,   # Each tree sees 80% of features (random)
    'min_child_weight': 3,     # Minimum sum of sample weights in a leaf
    'gamma': 0.1,              # Minimum improvement needed to make a split
    'reg_alpha': 0.1,          # L1 regularization (pushes small weights to zero)
    'reg_lambda': 1.0,         # L2 regularization (keeps all weights small)
    'scale_pos_weight': 1.5,   # Extra weight for the positive (phishing) class
    'tree_method': 'hist',     # Histogram-based algorithm — faster on large datasets
    'eval_metric': 'logloss'   # Use log loss to measure prediction quality
}
```

**Regularization (`reg_alpha`, `reg_lambda`):**
Regularization prevents **overfitting** — the model memorizing training data instead of
learning general patterns. It does this by penalizing models that are too complex.
Think of it as a "keep it simple" rule applied mathematically.

**Early Stopping:**
```python
# src/models/xgboost_model.py  (line 42-47)
if eval_set:
    self.model.fit(X, y, eval_set=[eval_set], verbose=False)
```
XGBoost checks the validation set after each tree is added. If performance stops improving
for many rounds, training stops — no point adding more trees that only overfit.
This is why the validation split is needed specifically for XGBoost.

---

## 6. The Full Training Orchestration — `src/models/trainer.py`

```python
# src/models/trainer.py  (line 37-97)
def train_all(self):
    # 1. Load parquet files → DataFrames
    df_train_raw = self.loader.load_data("data/processed/train.parquet")

    # 2. Align columns to master schema
    X_train_raw = df_train_raw.drop(['label', 'url'], axis=1, errors='ignore')
    X_train_raw = X_train_raw.reindex(columns=FEATURE_ORDER, fill_value=0)
    y_train_raw = df_train_raw['label']

    # 3. Fit imputer and scaler on training data ONLY, then apply to both
    X_train_processed = self.preprocessor.fit_transform(X_train_raw)
    X_test_processed  = self.preprocessor.transform(X_test_raw)  # No fit here!
    self.preprocessor.save("data/models/")  # Save imputer.joblib + scaler.joblib

    # 4. SMOTE on training set
    X_train_balanced, y_train_balanced = self.splitter.apply_smote(X_train_sub, y_train_sub)

    # 5. Train both models
    self.rf_model.fit(X_train_balanced, y_train_balanced)
    self.xgb_model.fit(X_train_balanced, y_train_balanced, eval_set=(X_val, y_val))

    # 6. Evaluate on test set
    self._evaluate_and_log(X_test_processed, y_test_raw)

    # 7. Save model files
    joblib.dump(self.rf_model.model, "data/models/random_forest.pkl")
    joblib.dump(self.xgb_model.model, "data/models/xgboost.pkl")
```

`joblib.dump` is Python's efficient serialization tool for large numpy arrays (better than `pickle`
for ML models). It compresses the trained model and saves it to disk. At server startup,
`joblib.load` reads those files back and the models are ready to predict instantly —
no retraining needed.

---

## 7. Evaluation Metrics — `src/models/trainer.py`

```python
# src/models/trainer.py  (line 112-117)
acc = accuracy_score(y_test, preds)    # % of correct predictions
f1  = f1_score(y_test, preds)          # Harmonic mean of precision & recall
auc = roc_auc_score(y_test, proba)     # Area Under the ROC Curve
```

**Why not just use accuracy?**
With imbalanced data, a model that always predicts "safe" achieves 90% accuracy but
catches zero phishing. Better metrics:

| Metric | What it measures | Good value |
|--------|-----------------|-----------|
| Accuracy | Overall correct predictions | >95% |
| F1 Score | Balance of precision (not crying wolf) vs recall (catching all phish) | >0.92 |
| AUC-ROC | How well the model ranks phishing above safe across all thresholds | >0.97 |

A high AUC means the model almost always assigns a higher phishing probability to actual
phishing URLs than to safe ones, regardless of where you set the decision threshold.
