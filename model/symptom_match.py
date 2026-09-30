"""Dataset-based symptom matching for sparse user input.

Ranks diseases by how well the user's symptoms match the actual rows in
data/dataset.csv. The score is a symptom-match score, NOT a calibrated
medical probability.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    from model.utils import normalize_symptom
except ImportError:  # when model/ itself is on sys.path
    from utils import normalize_symptom

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
FULL_WEIGHT = 0.6
COVERAGE_WEIGHT = 0.4


@lru_cache(maxsize=1)
def _load_profiles():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    disease_col = "Disease" if "Disease" in df.columns else df.columns[0]
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

    profiles = {}
    for _, row in df.iterrows():
        symptoms = frozenset(
            n
            for n in (
                normalize_symptom(s)
                for s in row[symptom_cols]
                if isinstance(s, str) and s.strip()
            )
            if n
        )
        profiles.setdefault(str(row[disease_col]).strip(), []).append(symptoms)
    return profiles


def rank_by_symptom_match(symptoms, top_n=10):
    """Rank diseases by dataset symptom match. top_n=None returns all.

    Each item: {"disease", "match_score" (0-100), "full_match_rows", "total_rows"}
    """
    user = {n for n in (normalize_symptom(s) for s in symptoms) if n}
    if not user:
        return []

    results = []
    for disease, rows in _load_profiles().items():
        total = len(rows)
        full = sum(1 for r in rows if user <= r)
        coverage = sum(len(user & r) / len(user) for r in rows) / total
        score = FULL_WEIGHT * (full / total) + COVERAGE_WEIGHT * coverage
        if full > 0:
            results.append(
                {
                    "disease": disease,
                    "match_score": round(score * 100, 1),
                    "full_match_rows": full,
                    "total_rows": total,
                }
            )

    results.sort(key=lambda d: (d["match_score"], d["full_match_rows"]), reverse=True)
    return results if top_n is None else results[:top_n]
