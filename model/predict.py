import joblib
import numpy as np
import pandas as pd
import json

model = joblib.load("model/medicheck_model.pkl")
le = joblib.load("model/label_encoder.pkl")
symptom_list = joblib.load("model/symptom_list.pkl")

desc_df = pd.read_csv("data/symptom_Description.csv")
desc_df.columns = desc_df.columns.str.strip()

precaution_df = pd.read_csv("data/symptom_precaution.csv")
precaution_df.columns = precaution_df.columns.str.strip()

severity_df = pd.read_csv("data/Symptom-severity.csv")
severity_df.columns = severity_df.columns.str.strip()
severity_df["Symptom"] = severity_df["Symptom"].str.strip().str.lower().str.replace(" ", "_")

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
    total = 0
    for s in symptoms:
        match = severity_df[severity_df["Symptom"] == s]
        if not match.empty:
            total += int(match["weight"].values[0])
    avg = total / len(symptoms) if symptoms else 0
    if avg >= 4:
        return "Urgent", "red"
    elif avg >= 2:
        return "Moderate", "orange"
    else:
        return "Mild", "green"

def predict_disease(symptoms):
    if not symptoms:
        return None

    # Build input vector
    input_vector = [1 if s in symptoms else 0 for s in symptom_list]
    input_array = np.array(input_vector).reshape(1, -1)

    # Get top 3 predictions with confidence
    proba = model.predict_proba(input_array)[0]
    top3_indices = np.argsort(proba)[::-1][:3]

    predictions = []
    for idx in top3_indices:
        disease = le.classes_[idx]
        confidence = round(proba[idx] * 100, 1)

        # Get description
        desc_match = desc_df[desc_df["Disease"].str.strip().str.lower() == disease.lower()]
        description = desc_match["Description"].values[0] if not desc_match.empty else "No description available."

        # Get precautions
        prec_match = precaution_df[precaution_df["Disease"].str.strip().str.lower() == disease.lower()]
        if not prec_match.empty:
            precautions = [prec_match.iloc[0][f"Precaution_{i}"] for i in range(1, 5) if pd.notna(prec_match.iloc[0][f"Precaution_{i}"])]
        else:
            precautions = []

        specialist = SPECIALIST_MAP.get(disease.lower(), "General Physician")
        severity, severity_color = get_severity(symptoms)

        predictions.append({
            "disease": disease.title(),
            "confidence": confidence,
            "description": description,
            "precautions": precautions,
            "specialist": specialist,
            "severity": severity,
            "severity_color": severity_color
        })

    return predictions