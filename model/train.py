"""
Train MediCheck's disease-prediction model.

Fixes vs. the original script:
- The dataset stores each row as a disease + up to 17 symptom slots
  (Symptom_1 ... Symptom_17), NOT as one-hot columns. The old script fed
  those raw string columns directly into scikit-learn, which cannot
  accept string features -> training crashed.
- This version melts the symptom slots into a single set of symptoms
  per row, then builds a proper binary (multi-hot) feature matrix:
  one column per unique symptom, 1 if the row has that symptom.
- Symptom names are normalized once, here, and the exact same
  normalization is reused in model/predict.py and
  model/symptom_extractor.py so severity lookups and predictions never
  drift out of sync again.

Run: python -m model.train   (from the project root)
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from utils import normalize_symptom  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_DIR = os.path.dirname(__file__)


def build_binary_matrix(df: pd.DataFrame):
    """Convert the 17 Symptom_N slot columns into one binary column per
    unique symptom (a proper multi-hot feature matrix for classification).
    """
    symptom_cols = [c for c in df.columns if c != "Disease"]

    # Collect the set of symptoms present in each row.
    rows_symptoms = []
    all_symptoms = set()
    for _, row in df.iterrows():
        present = set()
        for col in symptom_cols:
            val = row[col]
            if isinstance(val, str) and val.strip():
                s = normalize_symptom(val)
                if s:
                    present.add(s)
        rows_symptoms.append(present)
        all_symptoms |= present

    symptom_list = sorted(all_symptoms)

    X = pd.DataFrame(
        [[1 if s in present else 0 for s in symptom_list] for present in rows_symptoms],
        columns=symptom_list,
    )
    return X, symptom_list


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    X, symptom_list = build_binary_matrix(df)
    y = df["Disease"].str.strip()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Loaded {len(df)} rows, {len(symptom_list)} unique symptoms, {len(le.classes_)} diseases")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = round(acc * 100, 2)
        fitted[name] = model
        print(f"{name} accuracy: {results[name]}%")

    best_name = max(results, key=results.get)
    best_model = fitted[best_name]
    print(f"Best model: {best_name} ({results[best_name]}%)")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "medicheck_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(symptom_list, os.path.join(MODEL_DIR, "symptom_list.pkl"))

    with open(os.path.join(MODEL_DIR, "model_results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print("Saved medicheck_model.pkl, label_encoder.pkl, symptom_list.pkl, model_results.json")


if __name__ == "__main__":
    main()
