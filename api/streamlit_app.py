import streamlit as st
import sys
import os
import json
import sqlite3
from datetime import datetime

# ── Fix import path ───────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.symptom_extractor import extract_symptoms
from model.predict import predict_disease

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MediCheck",
    page_icon="🩺",
    layout="centered"
)

# ── Ensure DB folder exists ───────────────────────────────────
os.makedirs("data", exist_ok=True)

# ── Database setup ────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("data/queries.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT,
            top_prediction TEXT,
            severity TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(symptoms, top_prediction, severity):
    conn = sqlite3.connect("data/queries.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO queries (symptoms, top_prediction, severity, timestamp) VALUES (?, ?, ?, ?)",
        (", ".join(symptoms), top_prediction, severity,
         datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()

init_db()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediCheck")
    st.markdown("AI-powered symptom checker for Indian users.")
    st.markdown("---")

    age_group = st.selectbox(
        "Age group",
        ["Under 18", "18–35", "36–60", "Above 60"]
    )

    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Female", "Male", "Other"]
    )

    duration = st.selectbox(
        "How long have symptoms lasted?",
        ["Just started", "1–3 days", "4–7 days", "More than a week"]
    )

    st.markdown("---")
    st.caption("⚠️ Not a substitute for medical advice.")

# ── Header ────────────────────────────────────────────────────
st.title("🩺 MediCheck")
st.markdown("Describe symptoms in English, Hindi, or Kannada.")
st.markdown("---")

# ── Session State ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_symptoms" not in st.session_state:
    st.session_state.last_symptoms = []

if "predictions" not in st.session_state:
    st.session_state.predictions = None

# ── Follow-up logic ───────────────────────────────────────────
FOLLOWUP_QUESTIONS = {
    "high_fever": "How high is the fever? (mild / high / very high)",
    "headache": "Is it throbbing, constant, or one-sided?",
    "chest_pain": "Is it sharp, dull, or pressure-like?",
    "stomach_pain": "Before or after eating?",
    "cough": "Dry or with phlegm?",
    "fatigue": "Constant or after activity?",
    "dizziness": "Does it happen on standing?"
}

def get_followup(symptoms):
    for s in symptoms:
        if s in FOLLOWUP_QUESTIONS:
            return FOLLOWUP_QUESTIONS[s]
    return None

# ── Severity badge ────────────────────────────────────────────
def severity_badge(severity):
    return {
        "Urgent": "🔴",
        "Moderate": "🟡",
        "Mild": "🟢"
    }.get(severity, "🟢")

# ── Chat history display ──────────────────────────────────────
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# ── Main input ────────────────────────────────────────────────
user_input = st.chat_input("Type your symptoms here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    symptoms, lang = extract_symptoms(user_input)

    if not symptoms:
        response = "I couldn't identify any specific symptoms from your description. Could you try describing them differently? For example: 'I have fever, headache and body ache.'"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        st.session_state.last_symptoms = symptoms
        predictions = predict_disease(symptoms)

        if predictions:
            top = predictions[0]
            log_query(symptoms, top["disease"], top["severity"])

            lang_note = ""
            if lang != "en":
                lang_note = f"🌐 Input detected as non-English and translated automatically.\n\n"

            response = f"{lang_note}**Symptoms detected:** {', '.join([s.replace('_', ' ') for s in symptoms])}\n\n"
            response += f"**Severity:** {severity_badge(top['severity'])} {top['severity']}\n\n"
            response += f"---\n### Top 3 Possible Conditions\n\n"

            for i, pred in enumerate(predictions):
                bar = "█" * int(pred["confidence"] / 10) + "░" * (10 - int(pred["confidence"] / 10))
                response += f"**{i+1}. {pred['disease']}** — {pred['confidence']}%\n"
                response += f"`{bar}`\n\n"
                response += f"{pred['description']}\n\n"
                if pred["precautions"]:
                    response += f"**Precautions:** {', '.join(pred['precautions'])}\n\n"
                response += f"**See a:** {pred['specialist']}\n\n"
                if i < 2:
                    response += "---\n"

            followup = get_followup(symptoms)
            if followup:
                response += f"\n---\n💬 **Follow-up:** {followup}"
                st.session_state.awaiting_followup = True

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# ── PDF button — outside all conditionals ─────────────────────
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.last_symptoms:
        st.success(f"Last analysis: {', '.join([s.replace('_', ' ') for s in st.session_state.last_symptoms])}")
    else:
        st.info("Enter your symptoms above to get started.")

with col2:
     if len(st.session_state.last_symptoms) > 0:
        generate = st.button("📄 Get PDF Report", use_container_width=True)
        if generate:
            from model.pdf_report import generate_pdf
            with st.spinner("Generating your report..."):
                pdf_path = generate_pdf(
                    st.session_state.last_symptoms,
                    predict_disease(st.session_state.last_symptoms),
                    age_group,
                    gender,
                    duration
                )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Report",
                    data=f,
                    file_name="MediCheck_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
# ── Model comparison expander ─────────────────────────────────
with st.expander("📊 View model comparison"):
    try:
        with open("model/model_results.json") as f:
            results = json.load(f)
        for model_name, acc in results.items():
            st.markdown(f"**{model_name}**")
            st.progress(acc / 100)
            st.caption(f"{acc}% accuracy")
    except:
        st.caption("Run train.py first to see results.")