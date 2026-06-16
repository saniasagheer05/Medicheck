import joblib
import spacy
from deep_translator import GoogleTranslator
from langdetect import detect

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load symptom list (from dataset)
symptom_list = joblib.load("model/symptom_list.pkl")

# Normalize symptom list (convert to lowercase for matching)
symptom_list = [s.lower().strip() for s in symptom_list]

# Alias mapping (user-friendly → dataset format)
SYMPTOM_ALIASES = {
    "chest pain": "chest pain",
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
    "loose motion": "diarrhoea",
    "loose stools": "diarrhoea",
    "skin rash": "skin rash",
    "rash": "skin rash",
    "yellow skin": "yellowing of skin",
    "yellow eyes": "yellowing of eyes",
    "joint pain": "joint pain",
    "back pain": "back pain",
    "neck pain": "neck pain",
    "sore throat": "throat irritation",
    "frequent urination": "frequent urination",
    "burning urination": "burning micturition",
    "weight loss": "weight loss",
    "weight gain": "weight gain",
    "blurred vision": "blurred and distorted vision",
    "dizzy": "dizziness",
    "dry cough": "cough",
    "cold": "runny nose",
    "fever": "high fever",
    "mild fever": "mild fever",
    "low fever": "mild fever",
    "temperature": "high fever",
}

# Translate text if not English
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated, lang
        return text, "en"
    except:
        return text, "en"

# Main extraction function
def extract_symptoms(user_text):
    translated_text, detected_lang = translate_to_english(user_text)
    text_lower = translated_text.lower()

    matched_symptoms = []

    # Step 1: Alias matching
    for alias, symptom in SYMPTOM_ALIASES.items():
        if alias in text_lower:
            if symptom in symptom_list and symptom not in matched_symptoms:
                matched_symptoms.append(symptom)

    # Step 2: Direct dataset matching
    for symptom in symptom_list:
        if symptom in text_lower and symptom not in matched_symptoms:
            matched_symptoms.append(symptom)

    # Step 3: Token fallback (spaCy)
    doc = nlp(text_lower)
    for token in doc:
        if token.text in symptom_list and token.text not in matched_symptoms:
            matched_symptoms.append(token.text)

    return matched_symptoms, detected_lang