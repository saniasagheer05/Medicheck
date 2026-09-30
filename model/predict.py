import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import normalize_symptom, to_display, NON_SYMPTOM_ROWS  # noqa: E402
from symptom_match import rank_by_symptom_match  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

model = joblib.load(MODEL_DIR / "medicheck_model.pkl")
le = joblib.load(MODEL_DIR / "label_encoder.pkl")
symptom_list = joblib.load(MODEL_DIR / "symptom_list.pkl")

desc_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
desc_df.columns = desc_df.columns.str.strip()

precaution_df = pd.read_csv(DATA_DIR / "symptom_precaution.csv")
precaution_df.columns = precaution_df.columns.str.strip()

severity_df = pd.read_csv(DATA_DIR / "Symptom-severity.csv")
severity_df.columns = severity_df.columns.str.strip()
# FIX: previously this only stripped/lowercased and swapped spaces for
# underscores, but symptom_list.pkl was itself space-separated at the
# time (e.g. "skin rash"), so "skin rash" == "skin_rash" never matched
# and get_severity() silently returned "Mild" for every case. Both
# sides now go through the same normalize_symptom() used to build
# symptom_list.pkl in train.py, so they always agree.
severity_df["Symptom"] = severity_df["Symptom"].apply(normalize_symptom)
severity_df = severity_df[~severity_df["Symptom"].isin(NON_SYMPTOM_ROWS)]
SEVERITY_WEIGHTS = dict(zip(severity_df["Symptom"], severity_df["weight"]))

# Symptoms commonly associated with medical emergencies. Presence of any
# of these raises a hard risk flag regardless of the averaged severity
# score, since a single serious symptom shouldn't get diluted by several
# mild ones in an average.
HIGH_RISK_SYMPTOMS = {
    "chest_pain",
    "breathlessness",
    "coma",
    "altered_sensorium",
    "blood_in_sputum",
    "fluid_overload",
    "distention_of_abdomen",
    "unsteadiness",
    "slurred_speech",
    "loss_of_balance",
}

SPECIALIST_MAP = {
    "fungal infection": "Dermatologist",
    "allergy": "Allergist",
    "gerd": "Gastroenterologist",
    "chronic cholestasis": "Gastroenterologist",
    "drug reaction": "Dermatologist",
    "peptic ulcer diseae": "Gastroenterologist",
    "aids": "Infectious Disease Specialist",
    "diabetes": "Endocrinologist",
    "gastroenteritis": "Gastroenterologist",
    "bronchial asthma": "Pulmonologist",
    "hypertension": "Cardiologist",
    "migraine": "Neurologist",
    "cervical spondylosis": "Orthopedist",
    "paralysis (brain hemorrhage)": "Neurologist",
    "jaundice": "Hepatologist",
    "malaria": "General Physician",
    "chicken pox": "General Physician",
    "dengue": "General Physician",
    "typhoid": "General Physician",
    "hepatitis a": "Hepatologist",
    "hepatitis b": "Hepatologist",
    "hepatitis c": "Hepatologist",
    "hepatitis d": "Hepatologist",
    "hepatitis e": "Hepatologist",
    "alcoholic hepatitis": "Hepatologist",
    "tuberculosis": "Pulmonologist",
    "common cold": "General Physician",
    "pneumonia": "Pulmonologist",
    "dimorphic hemmorhoids(piles)": "Proctologist",
    "heart attack": "Cardiologist",
    "varicose veins": "Vascular Surgeon",
    "hypothyroidism": "Endocrinologist",
    "hyperthyroidism": "Endocrinologist",
    "hypoglycemia": "Endocrinologist",
    "osteoarthristis": "Orthopedist",
    "arthritis": "Orthopedist",
    "paroxysmal positional vertigo": "ENT Specialist",
    "acne": "Dermatologist",
    "urinary tract infection": "Urologist",
    "psoriasis": "Dermatologist",
    "impetigo": "Dermatologist",
}


def get_severity(symptoms):
    """Returns (label, color, risk_flag). symptoms must already be in
    canonical normalize_symptom() form.
    """
    symptoms = [normalize_symptom(s) for s in symptoms]
    total = sum(SEVERITY_WEIGHTS.get(s, 0) for s in symptoms)
    avg = total / len(symptoms) if symptoms else 0
    risk_flag = any(s in HIGH_RISK_SYMPTOMS for s in symptoms)

    if risk_flag or avg >= 4:
        return "Urgent", "red", risk_flag
    elif avg >= 2:
        return "Moderate", "orange", risk_flag
    else:
        return "Mild", "green", risk_flag


def predict_disease(symptoms, top_n=3):
    if not symptoms:
        return None

    symptoms = [normalize_symptom(s) for s in symptoms]

    # Rank by dataset symptom evidence instead of model.predict_proba: the
    # classifier was trained on full symptom profiles, so sparse user input
    # (unmentioned symptoms encoded as 0) is out-of-distribution for it.
    ranked = rank_by_symptom_match(symptoms, top_n=top_n)
    if not ranked:
        return None

    severity, severity_color, risk_flag = get_severity(symptoms)

    predictions = []
    for item in ranked:
        disease = item["disease"]
        confidence = item["match_score"]  # key kept for UI/PDF; it is a symptom-match score

        desc_match = desc_df[desc_df["Disease"].str.strip().str.lower() == disease.lower()]
        description = desc_match["Description"].values[0] if not desc_match.empty else "No description available."

        prec_match = precaution_df[precaution_df["Disease"].str.strip().str.lower() == disease.lower()]
        if not prec_match.empty:
            precautions = [
                prec_match.iloc[0][f"Precaution_{i}"]
                for i in range(1, 5)
                if pd.notna(prec_match.iloc[0][f"Precaution_{i}"])
            ]
        else:
            precautions = []

        specialist = SPECIALIST_MAP.get(disease.lower(), "General Physician")

        predictions.append({
            "disease": disease.title(),
            "confidence": confidence,
            "full_match_rows": item["full_match_rows"],
            "total_rows": item["total_rows"],
            "description": description,
            "precautions": precautions,
            "specialist": specialist,
            "severity": severity,
            "severity_color": severity_color,
            "risk_flag": risk_flag,
        })

    return predictions


def matched_symptoms_display(symptoms):
    """Human-readable version of the canonical symptoms, for showing the
    user what was actually understood from their input."""
    return [to_display(normalize_symptom(s)) for s in symptoms]
