import json
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.symptom_extractor import extract_symptoms  # noqa: E402
from model.predict import predict_disease  # noqa: E402

DB_PATH = os.path.join(BASE_DIR, "data", "queries.db")
RESULTS_PATH = os.path.join(BASE_DIR, "model", "model_results.json")

st.set_page_config(page_title="MediCheck", page_icon="🩺", layout="centered")


# ── DB ──────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO queries (symptoms, top_prediction, severity, timestamp) VALUES (?, ?, ?, ?)",
        (", ".join(symptoms), top_prediction, severity, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()


init_db()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediCheck")
    st.markdown("AI-powered symptom checker for Indian users.")
    st.markdown("---")
    age_group = st.selectbox("Age group", ["Under 18", "18–35", "36–60", "Above 60"])
    gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])
    duration = st.selectbox("How long have symptoms lasted?", ["Just started", "1–3 days", "4–7 days", "More than a week"])
    st.markdown("---")
    st.caption("⚠️ Not a substitute for medical advice. If this is an emergency, contact local emergency services immediately.")

# ── Header ───────────────────────────────────────────────────────
st.title("🩺 MediCheck")
st.markdown("Describe your symptoms in English, Hindi, or Kannada.")
st.markdown("---")

# ── Session state ────────────────────────────────────────────────
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_symptoms", [])
st.session_state.setdefault("last_predictions", None)

FOLLOWUP_QUESTIONS = {
    "high_fever": "How high is the fever? (mild / high / very high)",
    "headache": "Is it throbbing, constant, or one-sided?",
    "chest_pain": "Is it sharp, dull, or pressure-like?",
    "stomach_pain": "Before or after eating?",
    "cough": "Dry or with phlegm?",
    "fatigue": "Constant or after activity?",
    "dizziness": "Does it happen on standing?",
}


def get_followup(symptoms):
    for s in symptoms:
        if s in FOLLOWUP_QUESTIONS:
            return FOLLOWUP_QUESTIONS[s]
    return None


SEVERITY_BADGE = {"Urgent": "🔴", "Moderate": "🟡", "Mild": "🟢"}

# ── Replay chat history ──────────────────────────────────────────
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# ── Main input ───────────────────────────────────────────────────
user_input = st.chat_input("Type your symptoms here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    symptoms, lang = extract_symptoms(user_input)

    if not symptoms:
        response = (
            "I couldn't identify any specific symptoms from your description. "
            "Could you try describing them differently? For example: "
            "'I have fever, headache and body ache.'"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        predictions = predict_disease(symptoms)
        st.session_state.last_symptoms = symptoms
        st.session_state.last_predictions = predictions

        if predictions:
            top = predictions[0]
            log_query(symptoms, top["disease"], top["severity"])

            lang_note = "🌐 Input detected as non-English and translated automatically.\n\n" if lang != "en" else ""
            readable_symptoms = ", ".join(s.replace("_", " ") for s in symptoms)

            response = f"{lang_note}**Symptoms detected:** {readable_symptoms}\n\n"
            response += f"**Severity:** {SEVERITY_BADGE.get(top['severity'], '🟢')} {top['severity']}\n\n"
            if top.get("risk_flag"):
                response += "🚨 **One or more symptoms are commonly associated with medical emergencies. Please seek immediate medical attention.**\n\n"
            response += "---\n### Top Possible Conditions\n\n"

            for i, pred in enumerate(predictions):
                bar = "█" * int(pred["confidence"] / 10) + "░" * (10 - int(pred["confidence"] / 10))
                response += f"**{i+1}. {pred['disease']}** — {pred['confidence']}%\n"
                response += f"`{bar}`\n\n"
                response += f"{pred['description']}\n\n"
                if pred["precautions"]:
                    response += f"**Precautions:** {', '.join(pred['precautions'])}\n\n"
                response += f"**See a:** {pred['specialist']}\n\n"
                if i < len(predictions) - 1:
                    response += "---\n"

            followup = get_followup(symptoms)
            if followup:
                response += f"\n---\n💬 **Follow-up:** {followup}"

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

                # Confidence chart (high-value add: visual comparison of
                # the top candidate conditions instead of just text/%).
                chart_df = pd.DataFrame(
                    {"Condition": [p["disease"] for p in predictions], "Confidence %": [p["confidence"] for p in predictions]}
                ).set_index("Condition")
                st.bar_chart(chart_df)

                # Specialist cards (high-value add: scannable at a glance).
                cols = st.columns(len(predictions))
                for col, pred in zip(cols, predictions):
                    with col:
                        st.info(f"**{pred['disease']}**\n\n👨‍⚕️ {pred['specialist']}")

# ── Footer: last analysis + PDF export ───────────────────────────
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.last_symptoms:
        readable = ", ".join(s.replace("_", " ") for s in st.session_state.last_symptoms)
        st.success(f"Last analysis: {readable}")
    else:
        st.info("Enter your symptoms above to get started.")

with col2:
    if st.session_state.last_predictions:
        if st.button("📄 Get PDF Report", use_container_width=True):
            from model.pdf_report import generate_pdf

            with st.spinner("Generating your report..."):
                pdf_path = generate_pdf(
                    st.session_state.last_symptoms,
                    st.session_state.last_predictions,
                    age_group,
                    gender,
                    duration,
                )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Report",
                    data=f,
                    file_name="MediCheck_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# ── Model comparison ─────────────────────────────────────────────
with st.expander("📊 View model comparison"):
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        for model_name, acc in data.get("results", data).items():
            st.markdown(f"**{model_name}**")
            st.progress(acc / 100)
            st.caption(f"{acc}% accuracy")
        if "best_model" in data:
            st.caption(f"Selected for deployment: **{data['best_model']}**")
    except FileNotFoundError:
        st.caption("Run `python model/train.py` first to generate model_results.json.")
