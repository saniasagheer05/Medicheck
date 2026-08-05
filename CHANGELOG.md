# Changelog

## [Unreleased] — Portfolio-ready rewrite

### Fixed

- **`train.py` didn't run at all.** It called `df.applymap(...)` (removed
  in current pandas) and, even patched, fed raw string symptom columns
  directly into scikit-learn classifiers, which cannot accept string
  features. Rewritten to melt the 17 `Symptom_N` slot columns into a
  proper binary (multi-hot) feature matrix — one column per unique
  symptom — before training.
- **Missing model artifacts.** `medicheck_model.pkl` and
  `label_encoder.pkl` were not present in the repo, so `predict.py`
  couldn't load them. Retrained from the fixed `train.py` and committed
  all three artifacts (`medicheck_model.pkl`, `label_encoder.pkl`,
  `symptom_list.pkl`) plus `model_results.json`.
- **Severity always showed "Mild".** `symptom_list.pkl` stored symptoms
  space-separated (`"skin rash"`), while `Symptom-severity.csv` normalized
  to underscores (`"skin_rash"`). `get_severity()` compared them directly,
  so the lookup never matched and severity silently defaulted to the
  lowest tier regardless of input. Fixed by introducing a single shared
  `normalize_symptom()` in `model/utils.py`, used by `train.py`,
  `predict.py`, and `symptom_extractor.py` alike, so the format can't
  drift out of sync again. Covered by a regression test
  (`tests/test_predict.py::test_severity_is_not_always_mild`).
- **Two known dataset typos** corrected during normalization:
  `foul_smell_ofurine` → `foul_smell_of_urine`, and a stray non-symptom
  `"prognosis"` row dropped from the severity table.
- **`symptom_extractor.py` direct-matching step never matched anything**
  once `symptom_list` moved to underscore form, since user text is never
  underscore-separated. Fixed to compare against the human-readable form
  of each canonical symptom instead.
- Predictions now build the model input as a `DataFrame` with named
  columns instead of a bare `numpy` array, removing a scikit-learn
  "X does not have valid feature names" warning.

### Added

- `model/utils.py` — single source of truth for symptom normalization.
- **Risk flags**: symptoms commonly tied to medical emergencies (chest
  pain, breathlessness, altered sensorium, etc.) now force an "Urgent"
  severity and a visible warning, regardless of the averaged severity
  score — a single serious symptom is no longer diluted by several mild
  ones.
- **Confidence chart** and **specialist cards** in the Streamlit UI —
  visual comparison of the top candidate conditions instead of text only.
- **Optional FastAPI service** (`api/main.py`) exposing `/extract`,
  `/predict`, and `/predict-from-text` — the same pipeline usable outside
  Streamlit (mobile app, Postman demo, etc.).
- `tests/` — pytest suite covering symptom normalization, extraction,
  prediction shape, the severity regression, and artifact consistency
  (14 tests, all passing).
- `requirements.txt`, `.gitignore`, `.streamlit/config.toml` (theme).
- This `CHANGELOG.md` and an expanded `README.md` (architecture, model
  comparison table, setup/run/deploy instructions, known limitations).

### Changed

- **Single Streamlit entry point.** `api/streamlit_app.py` (full
  chat-style app) and `app/streamlit_app.py` (older, simpler duplicate)
  are consolidated into one file: `app/streamlit_app.py`. `api/` is now
  used only for the FastAPI service, matching its name.
- `model/pdf_report.py` now includes a risk-flag warning banner and uses
  path handling relative to the project root instead of the current
  working directory.

### Removed

- `create_symptoms.py` — its one job (building the unique symptom list)
  is now done inside `train.py`, using the same normalization as
  everything else, so there's one less script that can drift out of
  sync with the model.
- `app/streamlit_app.py` (old, simpler version) — superseded by the
  merged app described above.
- `api/streamlit_app.py` — moved/merged into `app/streamlit_app.py`.
