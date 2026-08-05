import os
import sys

import joblib
import spacy
from deep_translator import GoogleTranslator
from langdetect import detect

sys.path.insert(0, os.path.dirname(__file__))
from utils import normalize_symptom, to_display  # noqa: E402

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

nlp = spacy.load("en_core_web_sm")

# Canonical, underscore-form symptom names (e.g. "chest_pain"), matching
# what train.py used to build the feature matrix and what
# Symptom-severity.csv normalizes to. Longest-first so multi-word
# symptoms ("loss_of_appetite") are matched before shorter ones they
# happen to contain ("appetite").
symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))
symptom_list = sorted({normalize_symptom(s) for s in symptom_list}, key=len, reverse=True)

# User-friendly phrase -> canonical symptom. Written in plain English on
# both sides for readability; normalize_symptom() is applied when the
# alias is used, so this doesn't need to match symptom_list.pkl's exact
# on-disk format.
SYMPTOM_ALIASES = {
    "chest hurts": "chest pain",
    "stomach ache": "stomach pain",
    "tummy ache": "stomach pain",
    "throwing up": "vomiting",
    "feel like vomiting": "nausea",
    "can't sleep": "insomnia",
    "high temperature": "high fever",
    "running nose": "runny nose",
    "blocked nose": "congestion",
    "tired": "fatigue",
    "very tired": "fatigue",
    "exhausted": "fatigue",
    "head pain": "headache",
    "body pain": "muscle pain",
    "body ache": "muscle pain",
    "no appetite": "loss of appetite",
    "not hungry": "loss of appetite",
    "can't breathe": "breathlessness",
    "difficulty breathing": "breathlessness",
    "breathless": "breathlessness",
    "short of breath": "breathlessness",
    "loose motion": "diarrhoea",
    "loose stools": "diarrhoea",
    "rash": "skin rash",
    "yellow skin": "yellowing of skin",
    "yellow eyes": "yellowing of eyes",
    "sore throat": "throat irritation",
    "burning urination": "burning micturition",
    "blurred vision": "blurred and distorted vision",
    "dizzy": "dizziness",
    "dry cough": "cough",
    "cold": "runny nose",
    "fever": "high fever",
    "low fever": "mild fever",
    "temperature": "high fever",
}


def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated, lang
        return text, "en"
    except Exception:
        return text, "en"


def extract_symptoms(user_text):
    """Returns (matched_canonical_symptoms, detected_language).
    Matched symptoms are in canonical underscore form, ready to hand
    straight to predict.predict_disease().
    """
    translated_text, detected_lang = translate_to_english(user_text)
    text_lower = translated_text.lower()

    matched = []

    # Step 1: alias matching (natural phrases -> canonical symptom)
    for alias, symptom in SYMPTOM_ALIASES.items():
        if alias in text_lower:
            canonical = normalize_symptom(symptom)
            if canonical in symptom_list and canonical not in matched:
                matched.append(canonical)

    # Step 2: direct matching against the dataset's own symptom phrases.
    # FIX: symptom_list is stored in underscore form ("chest_pain"), but
    # user text never contains underscores -- comparing them directly
    # (as the original code did) meant this step could never match
    # anything. We compare against the human-readable form instead.
    for symptom in symptom_list:
        if to_display(symptom) in text_lower and symptom not in matched:
            matched.append(symptom)

    # Step 3: single-word token fallback (spaCy), for symptoms that are
    # already single words (e.g. "itching", "acidity", "vomiting").
    doc = nlp(text_lower)
    for token in doc:
        if token.text in symptom_list and token.text not in matched:
            matched.append(token.text)

    return matched, detected_lang
