import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.symptom_extractor import extract_symptoms  # noqa: E402
from model.utils import normalize_symptom, to_display  # noqa: E402


def test_normalize_symptom_handles_messy_input():
    assert normalize_symptom(" dischromic _patches") == "dischromic_patches"
    assert normalize_symptom("Skin Rash") == "skin_rash"
    assert normalize_symptom("foul_smell_ofurine") == "foul_smell_of_urine"


def test_to_display_reverses_normalization():
    assert to_display("skin_rash") == "skin rash"


def test_extract_symptoms_direct_phrase_match():
    symptoms, lang = extract_symptoms("I have a skin rash and itching")
    assert "skin_rash" in symptoms
    assert "itching" in symptoms
    assert lang == "en"


def test_extract_symptoms_alias_match():
    symptoms, _ = extract_symptoms("I can't breathe and my chest hurts")
    assert "breathlessness" in symptoms
    assert "chest_pain" in symptoms


def test_extract_symptoms_no_match_returns_empty_list():
    symptoms, _ = extract_symptoms("hello there, how are you today")
    assert symptoms == []
