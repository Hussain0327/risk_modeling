# Credit Risk Modeling

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white&style=flat-square) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=flat-square) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=flat-square) ![scikit-learn](https://img.shields.io/badge/sklearn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square) ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white&style=flat-square)

A machine learning pipeline for predicting loan defaults using LendingClub data. This project explores how financial institutions assess credit risk and which borrower characteristics drive default behavior.

## Overview

Banks face significant losses from loan defaults each year. This project investigates:

- What distinguishes borrowers who repay from those who default?
- Can we predict defaults before they occur?
- Which features carry the most predictive power?

## Dataset

**1.3M+ loans** from LendingClub (2007-2018) with 150+ features:

- Loan characteristics (amount, term, interest rate)
- Borrower profile (income, employment, home ownership)
- Credit history (utilization, delinquencies, inquiries)

### Target Distribution

![Loan Default Distribution](images/output.png)

The dataset exhibits a 20% default rate, presenting a class imbalance challenge addressed through weighted training.

## Analysis

### Interest Rate and Default Risk

Defaulted loans consistently show higher interest rates, indicating that risk-based pricing reflects genuine default probability.

![Interest Rate Analysis](images/output2.png)

### Income Distribution

Lower income quintiles show elevated default rates, though the relationship is more nuanced than expected. Borrowers across income levels demonstrate similar repayment patterns when other factors are controlled.

![Income Analysis](images/output3.png)

### Loan Grade Performance

Grade A loans default at 6%, while Grade G reaches nearly 50%. The internal grading system proves to be a reliable risk indicator.

![Default by Grade](images/output4.png)

### Feature Correlations

Interest rate (0.26 correlation) emerges as the strongest individual predictor. Income shows a weak negative correlation (-0.04) with default.

![Correlation Matrix](images/output5.png)

### Categorical Risk Factors

Renters default at higher rates than homeowners. Small business loans carry the highest risk at 30%.

![Category Analysis](images/output6.png)

## Model Results

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.948   |
| Random Forest       | 0.948   |

### ROC Curve Comparison

![ROC Comparison](images/05_01output.png)

Both models achieve near-identical performance, with the curves nearly overlapping.

### Confusion Matrix

![Confusion Matrix](images/05_output.png)

The model identifies 89% of defaults while maintaining acceptable false positive rates.

### Feature Importance

![Feature Importance](images/0502output.png)

FICO score range dominates feature importance, followed by interest rate and sub-grade. This aligns with industry practice where credit scores are primary risk indicators.

## Project Structure

```
risk_modeling/
├── images/                 # Visualizations and charts
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   └── 05_modeling.ipynb
├── src/
│   └── api/
│       └── risk_scorer.py  # Scoring API
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   └── trained/
└── requirements.txt
```

## Usage

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Notebooks

Execute notebooks 01-05 in order to reproduce the analysis.

### Scoring API

```python
from src.api.risk_scorer import CreditRiskScorer

scorer = CreditRiskScorer()
result = scorer.predict({
    "loan_amnt": 20000,
    "annual_inc": 80000,
    "int_rate": 12.5,
    "dti": 15.5,
    "grade": "B",
    "home_ownership": "RENT",
    "purpose": "debt_consolidation"
})

# Output:
# {
#     "risk_score": 42,
#     "decision": "MANUAL_REVIEW",
#     "default_probability": 0.42,
#     "confidence": "HIGH"
# }
```

## Key Takeaways

1. **Data leakage prevention** - Removed post-origination features that would not be available at loan decision time
2. **Class imbalance handling** - Applied balanced class weights to improve minority class detection
3. **Model interpretability** - Logistic regression performed comparably to ensemble methods while offering coefficient transparency
4. **FICO dominance** - Credit score ranges are by far the strongest predictors, validating industry reliance on credit bureaus

## Stack

pandas, numpy, scikit-learn, matplotlib, seaborn, kagglehub
