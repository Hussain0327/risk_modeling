"""
Credit Risk Scoring API

Returns risk score (0-100) and lending decision for a borrower profile.

Usage:
    from src.api.risk_scorer import CreditRiskScorer

    scorer = CreditRiskScorer()
    result = scorer.predict({
        "loan_amnt": 20000,
        "annual_inc": 80000,
        "int_rate": 12.5,
        ...
    })
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


class CreditRiskScorer:
    """Credit risk scoring engine for loan applications."""

    # Decision thresholds
    APPROVE_THRESHOLD = 30    # Risk score <= 30: Approve
    REVIEW_THRESHOLD = 60     # 30 < Risk score <= 60: Manual Review
    # Risk score > 60: Reject

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the scorer with trained model and scaler.

        Args:
            model_path: Path to saved model (default: models/trained/credit_risk_model.joblib)
            scaler_path: Path to saved scaler (default: models/scalers/standard_scaler.joblib)
        """
        base_path = Path(__file__).parent.parent.parent / "models"

        model_path = model_path or base_path / "trained/credit_risk_model.joblib"
        scaler_path = scaler_path or base_path / "scalers/standard_scaler.joblib"
        features_path = base_path / "trained/feature_names.joblib"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)

    def preprocess_input(self, borrower_profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert borrower profile to model-ready features.

        Applies the same feature engineering as training.
        """
        df = pd.DataFrame([borrower_profile])

        # Feature engineering (must match training)
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']

        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_to_income'] = (df['installment'] * 12) / df['annual_inc']

        if 'revol_util' in df.columns:
            df['high_utilization'] = (df['revol_util'] > 80).astype(int)

        # One-hot encode categoricals
        categorical_cols = ['term', 'grade', 'home_ownership', 'verification_status',
                           'purpose', 'application_type', 'initial_list_status',
                           'dti_risk', 'income_category']
        existing_cats = [c for c in categorical_cols if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

        # Align columns with training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Select only training features in correct order
        df = df[self.feature_names]

        return df

    def calculate_risk_score(self, default_probability: float) -> int:
        """Convert default probability to 0-100 risk score."""
        risk_score = int(default_probability * 100)
        return min(max(risk_score, 0), 100)

    def get_decision(self, risk_score: int) -> str:
        """Determine lending decision based on risk score."""
        if risk_score <= self.APPROVE_THRESHOLD:
            return "APPROVE"
        elif risk_score <= self.REVIEW_THRESHOLD:
            return "MANUAL_REVIEW"
        else:
            return "REJECT"

    def predict(self, borrower_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a loan application.

        Args:
            borrower_profile: Dictionary with borrower information.
                Required keys depend on trained model features.
                Common keys: loan_amnt, annual_inc, int_rate, dti,
                            grade, home_ownership, purpose, etc.

        Returns:
            Dictionary with:
                - risk_score: 0-100 (higher = riskier)
                - decision: APPROVE, MANUAL_REVIEW, or REJECT
                - default_probability: Raw probability of default
                - confidence: HIGH, MEDIUM, or LOW
        """
        # Preprocess
        X = self.preprocess_input(borrower_profile)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        default_prob = self.model.predict_proba(X_scaled)[0, 1]

        # Calculate score and decision
        risk_score = self.calculate_risk_score(default_prob)
        decision = self.get_decision(risk_score)

        # Confidence based on how far from decision boundary (0.5)
        prob_distance = abs(default_prob - 0.5)
        if prob_distance > 0.3:
            confidence = "HIGH"
        elif prob_distance > 0.15:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "risk_score": risk_score,
            "decision": decision,
            "default_probability": round(default_prob, 4),
            "confidence": confidence
        }


def main():
    """Example usage of the Credit Risk Scorer."""

    # Example borrower profile
    borrower = {
        "loan_amnt": 20000,
        "term": " 36 months",
        "int_rate": 12.5,
        "installment": 665.0,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 80000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 15.5,
        "open_acc": 8,
        "revol_bal": 12000,
        "revol_util": 45.0,
        "total_acc": 15
    }

    try:
        scorer = CreditRiskScorer()
        result = scorer.predict(borrower)

        print("=" * 50)
        print("CREDIT RISK ASSESSMENT")
        print("=" * 50)
        print(f"Risk Score:          {result['risk_score']}/100")
        print(f"Default Probability: {result['default_probability']:.2%}")
        print(f"Decision:            {result['decision']}")
        print(f"Confidence:          {result['confidence']}")
        print("=" * 50)

    except FileNotFoundError:
        print("Error: Model files not found.")
        print("Please run notebooks 01-05 first to train and save the model.")


if __name__ == "__main__":
    main()
