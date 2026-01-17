# Bug and Inconsistency Report
# Credit Risk Modeling Project

**Date:** December 25, 2025  
**Analysis Type:** Comprehensive Code Review  
**Status:** FINDINGS ONLY - NO FIXES APPLIED

---

## Executive Summary

This report documents **14 bugs and inconsistencies** found in the credit risk modeling codebase. The issues range from critical bugs that will cause runtime failures to minor code quality concerns.

**Severity Breakdown:**
- **CRITICAL:** 5 bugs (will cause failures)
- **HIGH:** 1 bug (causes incorrect behavior)
- **MEDIUM:** 3 bugs (edge case issues)
- **LOW:** 3 issues (code quality)
- **INFO:** 2 issues (best practices)

---

## Critical Bugs (Must Fix)

### 1. Missing Model File Error Handling
**Location:** `src/api/risk_scorer.py`, lines 47-49  
**Severity:** CRITICAL  
**Type:** ERROR_HANDLING

**Description:**  
The `__init__` method loads model files using `joblib.load()` without any error handling. If model files don't exist, the code will crash with an unhelpful `FileNotFoundError`.

**Code:**
```python
self.model = joblib.load(model_path)
self.scaler = joblib.load(scaler_path)
self.feature_names = joblib.load(features_path)
```

**Impact:**  
- API crashes on initialization if model files missing
- No helpful error message for users
- Makes debugging difficult

**Recommendation:**  
Wrap in try-except block and provide clear error message directing users to run training notebooks first.

---

### 2. Division by Zero - loan_to_income
**Location:** `src/api/risk_scorer.py`, line 61  
**Severity:** CRITICAL  
**Type:** DIVISION_BY_ZERO

**Description:**  
Calculates `loan_to_income` by dividing by `annual_inc` without checking if it's zero or null.

**Code:**
```python
if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
```

**Impact:**  
- Runtime error if `annual_inc` is 0
- Will return `inf` if not handled, corrupting predictions

**Test Case:**
```python
borrower = {"loan_amnt": 10000, "annual_inc": 0}
# This will create inf value
```

**Recommendation:**  
Add validation: `df['annual_inc'].replace(0, np.nan)` or check before division.

---

### 3. Division by Zero - payment_to_income
**Location:** `src/api/risk_scorer.py`, line 64  
**Severity:** CRITICAL  
**Type:** DIVISION_BY_ZERO

**Description:**  
Same issue as Bug #2 but for `payment_to_income` calculation.

**Code:**
```python
if 'installment' in df.columns and 'annual_inc' in df.columns:
    df['payment_to_income'] = (df['installment'] * 12) / df['annual_inc']
```

**Impact:**  
Same as Bug #2 - will create `inf` values.

---

### 4. Missing Log Transformations
**Location:** `src/api/risk_scorer.py`, `preprocess_input` method  
**Severity:** CRITICAL  
**Type:** MISSING_FEATURES

**Description:**  
Training notebook (03_feature_engineering.ipynb) creates `log_annual_inc` and `log_revol_bal` using `np.log1p()`, but the API does NOT create these features.

**Training Code (Notebook 03, Cell 9):**
```python
if "annual_inc" in df.columns:
    df["log_annual_inc"] = np.log1p(df["annual_inc"])

if "revol_bal" in df.columns:
    df["log_revol_bal"] = np.log1p(df["revol_bal"])
```

**API Code:**  
*These transformations are completely missing*

**Impact:**  
- Model expects `log_annual_inc` and `log_revol_bal` columns
- API will fail to predict because these columns won't exist
- Feature alignment with training data will fail
- **This bug makes the API completely non-functional**

**Recommendation:**  
Add log transformations to `preprocess_input` method before one-hot encoding.

---

### 5. Missing Categorical Feature Generation
**Location:** `src/api/risk_scorer.py`, `preprocess_input` method  
**Severity:** CRITICAL  
**Type:** MISSING_FEATURE_GENERATION

**Description:**  
The API lists `'dti_risk'` and `'income_category'` in `categorical_cols` (line 70-72) but never creates them. These are derived features created in training.

**Training Code (Notebook 03):**
```python
# DTI Risk Category
df["dti_risk"] = pd.cut(
    df["dti"],
    bins=[-np.inf, 10, 20, 35, np.inf],
    labels=["low", "moderate", "high", "very_high"],
)

# Income Category
df["income_category"] = pd.cut(
    df["annual_inc"],
    bins=[0, 30000, 60000, 100000, 200000, np.inf],
    labels=["low", "lower_middle", "middle", "upper_middle", "high"],
)
```

**API Code:**  
*These features are never created, just expected in categorical_cols*

**Impact:**  
- Users must manually create these categories
- Breaks API usability
- Documentation doesn't explain this requirement
- API will fail when trying to one-hot encode non-existent columns

**Recommendation:**  
Add `pd.cut()` logic to create these categorical features in `preprocess_input`.

---

## High Severity Bugs

### 6. Train-Serve Skew - Feature Clipping
**Location:** `src/api/risk_scorer.py` vs `notebooks/03_feature_engineering.ipynb`  
**Severity:** HIGH  
**Type:** TRAIN_SERVE_SKEW

**Description:**  
Training notebook clips `loan_to_income` and `payment_to_income` to 99th percentile, but API does not apply clipping.

**Training Code (Notebook 03):**
```python
df["loan_to_income"] = df["loan_to_income"].clip(
    upper=df["loan_to_income"].quantile(0.99)
)
df["payment_to_income"] = df["payment_to_income"].clip(
    upper=df["payment_to_income"].quantile(0.99)
)
```

**API Code:**  
*No clipping applied*

**Impact:**  
- Extreme values handled differently in training vs prediction
- Model trained on clipped values will make poor predictions on extreme inputs
- Degrades model performance on edge cases

**Recommendation:**  
Either:
1. Save clip thresholds during training and apply in API
2. Remove clipping from training (less recommended)

---

## Medium Severity Bugs

### 7 & 8. Division by Zero in Training Notebooks
**Location:** `notebooks/03_feature_engineering.ipynb`, Cells 4 & 5  
**Severity:** MEDIUM  
**Type:** DIVISION_BY_ZERO

**Description:**  
Same division by zero issues as Bugs #2 and #3, but in training notebooks.

**Impact:**  
- If training data contains `annual_inc = 0`, will create `inf` values
- Less critical than API bugs since training data has likely been pre-cleaned
- Still should be handled for robustness

---

### 9. Employment Length Edge Case Handling
**Location:** `notebooks/02_data_cleaning.ipynb`, Cell 11  
**Severity:** MEDIUM  
**Type:** INCOMPLETE_PARSING

**Description:**  
Employment length parsing uses `str.extract(r"(\d+)")` which doesn't handle "< 1 year" correctly.

**Code:**
```python
df["emp_length_numeric"] = df["emp_length"].str.extract(r"(\d+)").astype(float)
df["emp_length_numeric"] = df["emp_length_numeric"].fillna(0)
```

**Impact:**  
- "< 1 year" becomes NaN, then 0.0
- Treats "< 1 year" same as "no employment" 
- Should probably be 0.5 or handled specially

**Edge Cases:**
- "< 1 year" → 0.0 (misleading)
- "10+ years" → 10.0 (OK, but loses "+" information)
- "n/a" → 0.0 (OK for missing data)

---

### 10. Term Field Formatting Inconsistency
**Location:** `src/api/risk_scorer.py`, line 154  
**Severity:** MEDIUM  
**Type:** DATA_INCONSISTENCY

**Description:**  
Example in `main()` uses `"term": " 36 months"` with leading space, which must match training data exactly for one-hot encoding.

**Code:**
```python
borrower = {
    "term": " 36 months",  # Leading space
    ...
}
```

**Impact:**  
- If training data has `" 36 months"` but user provides `"36 months"`, one-hot encoding will fail
- Creates column name `term_ 36 months` vs `term_36 months`
- Prediction will use wrong features
- No validation or string normalization

**Recommendation:**  
Strip whitespace in preprocessing: `df['term'] = df['term'].str.strip()`

---

## Low Severity Issues

### 11. Global Warning Suppression
**Location:** All notebooks, first cell  
**Severity:** LOW  
**Type:** CODE_SMELL

**Description:**  
All notebooks use `warnings.filterwarnings("ignore")` which suppresses ALL warnings globally.

**Code:**
```python
warnings.filterwarnings("ignore")
```

**Impact:**  
- Hides potentially important warnings
- Makes debugging harder
- May hide deprecation warnings for future library updates

**Recommendation:**  
Suppress specific warning categories:
```python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

---

### 12. Hardcoded Relative Paths
**Location:** All notebooks  
**Severity:** LOW  
**Type:** CODE_SMELL

**Description:**  
Notebooks use hardcoded relative paths like `"../data/raw"`.

**Code:**
```python
DATA_DIR = Path("../data/raw")
```

**Impact:**  
- Breaks if notebooks run from different directories
- Not portable across different environments
- Makes automation harder

**Recommendation:**  
Use project root detection or environment variables.

---

### 13. README Example Incomplete
**Location:** `README.md`, lines 120-141  
**Severity:** LOW  
**Type:** DOCUMENTATION_ISSUE

**Description:**  
README example is incomplete compared to the actual API requirements.

**README Example:**
```python
result = scorer.predict({
    "loan_amnt": 20000,
    "annual_inc": 80000,
    "int_rate": 12.5,
    "dti": 15.5,
    "grade": "B",
    "home_ownership": "RENT",
    "purpose": "debt_consolidation"
})
```

**Actual API Example (from risk_scorer.py main()):**
```python
borrower = {
    "loan_amnt": 20000,
    "term": " 36 months",        # Missing in README
    "int_rate": 12.5,
    "installment": 665.0,        # Missing in README
    "grade": "B",
    "sub_grade": "B3",          # Missing in README
    "emp_length": "5 years",    # Missing in README
    ...
}
```

**Impact:**  
- Users following README will get prediction errors
- Missing required fields not documented

---

## Info Level Issues

### 14. Missing Input Validation
**Location:** `src/api/risk_scorer.py`, `predict` method  
**Severity:** INFO  
**Type:** MISSING_VALIDATION

**Description:**  
The `predict()` method doesn't validate that required fields are present before processing.

**Impact:**  
- Cryptic errors if required fields missing
- Poor user experience
- Hard to debug

**Recommendation:**  
Add validation at start of `predict()`:
```python
required_fields = ['loan_amnt', 'annual_inc', 'int_rate', ...]
missing = [f for f in required_fields if f not in borrower_profile]
if missing:
    raise ValueError(f"Missing required fields: {missing}")
```

---

## Additional Observations

### Positive Findings ✓

1. **No security vulnerabilities** - No use of `eval()`, `exec()`, or unsafe operations
2. **Good version pinning** - All requirements specify minimum versions
3. **Consistent random_state** - All models use `random_state=42`
4. **Class imbalance handling** - All models use `class_weight="balanced"`
5. **No unused imports** - All imports are utilized
6. **Clean syntax** - No syntax errors in Python files

### Architecture Concerns

1. **No model versioning** - No way to track which model version is deployed
2. **No input sanitization** - User input not validated for security
3. **No logging** - No logging for debugging or monitoring
4. **No error metrics** - API doesn't return confidence intervals or error estimates

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Total Issues** | 14 |
| Critical | 5 |
| High | 1 |
| Medium | 3 |
| Low | 3 |
| Info | 2 |

**Top 3 Most Critical Issues:**
1. Missing log transformations (makes API non-functional)
2. Missing categorical feature generation (breaks predictions)  
3. Division by zero errors (causes runtime failures)

---

## Reproduction Steps

To reproduce the bugs:

```bash
# Bug #4 - Missing log transformations
python3 << EOF
from src.api.risk_scorer import CreditRiskScorer
scorer = CreditRiskScorer()  # Assuming models exist
result = scorer.predict({
    "loan_amnt": 10000,
    "annual_inc": 50000,
    # ... other fields
})
# Will fail because log_annual_inc and log_revol_bal missing
EOF

# Bug #2 - Division by zero
python3 << EOF
from src.api.risk_scorer import CreditRiskScorer
scorer = CreditRiskScorer()
result = scorer.predict({
    "loan_amnt": 10000,
    "annual_inc": 0,  # Zero income
    # ... other fields
})
# Will create inf values
EOF
```

---

## Files Analyzed

1. `src/api/risk_scorer.py` - Main API implementation
2. `src/api/__init__.py` - API module init
3. `src/__init__.py` - Package init
4. `notebooks/01_data_acquisition.ipynb` - Data download
5. `notebooks/02_data_cleaning.ipynb` - Data cleaning
6. `notebooks/03_feature_engineering.ipynb` - Feature creation
7. `notebooks/04_eda.ipynb` - Exploratory analysis
8. `notebooks/05_modeling.ipynb` - Model training
9. `README.md` - Documentation
10. `requirements.txt` - Dependencies

---

## Notes

- This is a **FINDINGS ONLY** report
- **NO FIXES have been applied** to the codebase
- All bugs remain in their current state
- This report is for informational purposes only
- Prioritize fixing CRITICAL bugs first

---

**End of Report**
