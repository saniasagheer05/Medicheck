"""
Shared helpers used by train.py, predict.py and symptom_extractor.py.

Keeping normalization in exactly one place is the fix for the original
"severity always shows Mild" bug: symptom_list.pkl (space-separated,
built by the old create_symptoms.py) and Symptom-severity.csv
(underscore-separated) never matched, so get_severity() never found a
row and silently defaulted to the lowest severity every time. Every
module now normalizes through this same function.
"""
import re

# The source Kaggle dataset has a couple of known typos/artifacts that
# don't match across its own files. We correct them once, here, instead
# of patching around them in multiple places.
KNOWN_SYMPTOM_FIXES = {
    "foul_smell_ofurine": "foul_smell_of_urine",
}

# "prognosis" is a stray header leak in Symptom-severity.csv, not a
# real symptom -- it never appears in dataset.csv, so it's dropped
# wherever the severity table is loaded.
NON_SYMPTOM_ROWS = {"prognosis"}


def normalize_symptom(value: str) -> str:
    """Canonical symptom format: lowercase, single underscores, no stray
    leading/trailing whitespace or underscores.
    e.g. ' dischromic _patches' -> 'dischromic_patches'
    """
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return KNOWN_SYMPTOM_FIXES.get(value, value)


def to_display(symptom: str) -> str:
    """Canonical underscore form -> human-readable, for showing in the UI."""
    return symptom.replace("_", " ")
