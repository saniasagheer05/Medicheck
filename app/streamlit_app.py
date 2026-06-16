import streamlit as st
from model.symptom_extractor import extract_symptoms

st.set_page_config(page_title="MediCheck", page_icon="🩺")

st.title("🩺 MediCheck - AI Symptom Checker")

st.write("Enter your symptoms in simple language")

user_input = st.text_input("Example: I have fever and headache")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms")
    else:
        symptoms, lang = extract_symptoms(user_input)

        st.subheader("Detected Symptoms:")
        st.write(symptoms)

        st.subheader("Detected Language:")
        st.write(lang)