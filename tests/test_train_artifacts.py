import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")


def test_artifacts_exist():
    for fname in ["medicheck_model.pkl", "label_encoder.pkl", "symptom_list.pkl"]:
        assert os.path.exists(os.path.join(MODEL_DIR, fname)), f"Missing {fname} — run `python model/train.py` first."


def test_symptom_list_matches_model_feature_count():
    model = joblib.load(os.path.join(MODEL_DIR, "medicheck_model.pkl"))
    symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))
    assert model.n_features_in_ == len(symptom_list)


def test_symptom_list_is_underscore_normalized():
    """Regression test for the original bug: symptom_list.pkl must use
    the same underscore format as Symptom-severity.csv, or severity
    lookups silently fail and always return the lowest severity."""
    symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))
    assert all(" " not in s for s in symptom_list)
    assert all(s == s.lower() for s in symptom_list)


def test_label_encoder_has_diseases():
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    assert len(le.classes_) > 0
