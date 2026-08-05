import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import normalize_symptom, to_display, NON_SYMPTOM_ROWS  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

model = joblib.load(os.path.join(MODEL_DIR, "medicheck_model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))

desc_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
desc_df.columns = desc_df.columns.str.strip()

precaution_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
precaution_df.columns = precaution_df.columns.str.strip()

severity_df = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
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

    input_vector = [1 if s in symptoms else 0 for s in symptom_list]
    input_df = pd.DataFrame([input_vector], columns=symptom_list)

    proba = model.predict_proba(input_df)[0]
    top_indices = np.argsort(proba)[::-1][:top_n]

    severity, severity_color, risk_flag = get_severity(symptoms)

    predictions = []
    for idx in top_indices:
        disease = le.classes_[idx]
        confidence = round(proba[idx] * 100, 1)

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
