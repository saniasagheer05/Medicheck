from model.symptom_match import rank_by_symptom_match

MUST_APPEAR = {"dengue", "typhoid", "malaria", "chicken pox", "common cold"}
MUST_NOT_APPEAR = {"aids", "paralysis (brain hemorrhage)"}


def _ranked_by_name(symptoms):
    return {r["disease"].lower(): r for r in rank_by_symptom_match(symptoms, top_n=None)}


def test_fever_headache_excludes_diseases_without_evidence():
    ranked = _ranked_by_name(["high_fever", "headache"])
    for disease in MUST_NOT_APPEAR:
        assert disease not in ranked


def test_fever_headache_includes_diseases_with_exact_rows():
    ranked = _ranked_by_name(["high_fever", "headache"])
    for disease in MUST_APPEAR:
        assert disease in ranked
        assert ranked[disease]["full_match_rows"] > 0


def test_scores_are_bounded_and_sorted():
    ranked = rank_by_symptom_match(["high_fever", "headache"], top_n=None)
    scores = [r["match_score"] for r in ranked]
    assert all(0 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_empty_input():
    assert rank_by_symptom_match([]) == []
