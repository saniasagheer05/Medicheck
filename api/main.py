"""
Optional REST API for MediCheck, built on the same model/ pipeline used
by the Streamlit app. Useful if you want to call predictions from a
mobile app, a different frontend, or curl/Postman for a demo.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict_disease  # noqa: E402
from model.symptom_extractor import extract_symptoms  # noqa: E402

app = FastAPI(
    title="MediCheck API",
    description="AI-powered symptom checker — predicts likely conditions from reported symptoms.",
    version="1.0.0",
)


class TextRequest(BaseModel):
    text: str


class SymptomsRequest(BaseModel):
    symptoms: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(payload: TextRequest):
    """Free-text symptom description -> canonical symptom list."""
    symptoms, lang = extract_symptoms(payload.text)
    return {"symptoms": symptoms, "detected_language": lang}


@app.post("/predict")
def predict(payload: SymptomsRequest):
    """Canonical symptom list (e.g. from /extract, or your own UI) -> predictions."""
    predictions = predict_disease(payload.symptoms)
    if not predictions:
        return {"predictions": []}
    return {"predictions": predictions}


@app.post("/predict-from-text")
def predict_from_text(payload: TextRequest):
    """Convenience endpoint: free text straight to predictions."""
    symptoms, lang = extract_symptoms(payload.text)
    predictions = predict_disease(symptoms)
    return {
        "detected_language": lang,
        "symptoms": symptoms,
        "predictions": predictions or [],
    }
