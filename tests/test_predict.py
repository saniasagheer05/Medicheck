import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import get_severity, predict_disease  # noqa: E402


def test_predict_disease_returns_ranked_predictions():
    preds = predict_disease(["itching", "skin_rash"])
    assert preds is not None
    assert len(preds) == 3
    confidences = [p["confidence"] for p in preds]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_disease_empty_input_returns_none():
    assert predict_disease([]) is None


def test_predict_disease_result_has_expected_fields():
    preds = predict_disease(["high_fever", "headache"])
    required = {"disease", "confidence", "description", "precautions", "specialist", "severity", "severity_color", "risk_flag"}
    assert required.issubset(preds[0].keys())


def test_severity_is_not_always_mild():
    """Regression test for the original bug: before the fix, get_severity()
    always returned 'Mild' because symptom_list.pkl (space-separated) never
    matched Symptom-severity.csv (underscore-separated)."""
    severity, _, _ = get_severity(["high_fever", "vomiting", "yellowish_skin"])
    assert severity != "Mild"


def test_high_risk_symptom_sets_risk_flag():
    _, _, risk_flag = get_severity(["chest_pain"])
    assert risk_flag is True
