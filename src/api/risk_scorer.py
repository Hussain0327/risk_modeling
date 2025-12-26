"""
Credit Risk Scoring API

Returns risk score (0-100) and lending decision for a borrower profile.

Usage:
    from src.api.risk_scorer import CreditRiskScorer

    scorer = CreditRiskScorer()
    result = scorer.predict({
        "loan_amnt": 20000,
        "annual_inc": 80000,
        "dti": 15.5,
        "installment": 665.0,
        "term": " 36 months",
        "grade": "B",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "revol_bal": 12000,
        "revol_util": 45.0,
    })
    # Returns: {"risk_score": 45, "decision": "MANUAL_REVIEW", ...}
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

    # Feature clipping thresholds (from training 99th percentile)
    LOAN_TO_INCOME_CLIP = 0.5
    PAYMENT_TO_INCOME_CLIP = 0.2

    # Required fields for prediction
    REQUIRED_FIELDS = ['loan_amnt', 'annual_inc', 'dti', 'installment']

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

        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run notebooks 01-05 to train and save the model."
            )

        try:
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}\n"
                "Run notebooks 01-05 to train and save the scaler."
            )

        try:
            self.feature_names = joblib.load(features_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature names file not found: {features_path}\n"
                "Run notebooks 01-05 to save feature names."
            )

    def preprocess_input(self, borrower_profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert borrower profile to model-ready features.

        Applies the same feature engineering as training.
        """
        df = pd.DataFrame([borrower_profile])

        # === RATIO FEATURES (with zero protection and clipping) ===

        # Loan-to-Income Ratio
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = np.where(
                df['annual_inc'] > 0,
                df['loan_amnt'] / df['annual_inc'],
                self.LOAN_TO_INCOME_CLIP  # Max value for zero income
            )
            df['loan_to_income'] = df['loan_to_income'].clip(upper=self.LOAN_TO_INCOME_CLIP)

        # Payment-to-Income Ratio
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_to_income'] = np.where(
                df['annual_inc'] > 0,
                (df['installment'] * 12) / df['annual_inc'],
                self.PAYMENT_TO_INCOME_CLIP  # Max value for zero income
            )
            df['payment_to_income'] = df['payment_to_income'].clip(upper=self.PAYMENT_TO_INCOME_CLIP)

        # === BINARY FLAGS ===

        if 'revol_util' in df.columns:
            df['high_utilization'] = (df['revol_util'] > 80).astype(int)

        # === LOG TRANSFORMATIONS (must match training notebook cell-9) ===

        if 'annual_inc' in df.columns:
            df['log_annual_inc'] = np.log1p(df['annual_inc'])

        if 'revol_bal' in df.columns:
            df['log_revol_bal'] = np.log1p(df['revol_bal'])

        # === CATEGORICAL BINNING (must match training notebook cells 7-8) ===

        # DTI Risk Category
        if 'dti' in df.columns:
            df['dti_risk'] = pd.cut(
                df['dti'],
                bins=[-np.inf, 10, 20, 35, np.inf],
                labels=['low', 'moderate', 'high', 'very_high']
            )

        # Income Category
        if 'annual_inc' in df.columns:
            df['income_category'] = pd.cut(
                df['annual_inc'],
                bins=[0, 30000, 60000, 100000, 200000, np.inf],
                labels=['low', 'lower_middle', 'middle', 'upper_middle', 'high']
            )

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

    def validate_input(self, borrower_profile: Dict[str, Any]) -> None:
        """
        Validate borrower profile has required fields with valid values.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        missing_fields = []
        invalid_fields = []

        for field in self.REQUIRED_FIELDS:
            if field not in borrower_profile:
                missing_fields.append(field)
            elif borrower_profile[field] is None:
                invalid_fields.append(f"{field} (cannot be None)")

        # Validate numeric ranges
        if 'loan_amnt' in borrower_profile:
            if borrower_profile['loan_amnt'] <= 0:
                invalid_fields.append("loan_amnt (must be positive)")

        if 'annual_inc' in borrower_profile:
            if borrower_profile['annual_inc'] < 0:
                invalid_fields.append("annual_inc (cannot be negative)")

        if 'dti' in borrower_profile:
            dti = borrower_profile['dti']
            if dti < 0:
                invalid_fields.append("dti (cannot be negative)")

        error_messages = []
        if missing_fields:
            error_messages.append(f"Missing required fields: {', '.join(missing_fields)}")
        if invalid_fields:
            error_messages.append(f"Invalid field values: {', '.join(invalid_fields)}")

        if error_messages:
            raise ValueError("\n".join(error_messages))

    def predict(self, borrower_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a loan application.

        Args:
            borrower_profile: Dictionary with borrower information.
                Required: loan_amnt, annual_inc, dti, installment
                Recommended: term, int_rate, grade, home_ownership,
                            purpose, revol_bal, revol_util

        Returns:
            Dictionary with:
                - risk_score: 0-100 (higher = riskier)
                - decision: APPROVE (<=30), MANUAL_REVIEW (31-60), REJECT (>60)
                - default_probability: Raw probability of default
                - confidence: HIGH, MEDIUM, or LOW

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Validate input
        self.validate_input(borrower_profile)

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
