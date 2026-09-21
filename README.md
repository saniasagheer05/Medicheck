<div align="center">

# 🩺 MediCheck

### AI-Powered Multilingual Symptom Checker

**Describe your symptoms in English, Hindi, or Kannada — get likely conditions, severity triage, and a specialist recommendation.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Classifier-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Optional%20REST%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Educational%2FPortfolio-lightgrey.svg)](#license)

[Screenshots](#-screenshots) · [How It Works](#-how-it-works) · [Model](#-model) · [Setup](#-setup--run-locally) · [Deploy](#-deploying-to-streamlit-cloud)

</div>

---

## 📖 Overview

**MediCheck** takes a free-text description of symptoms and returns likely conditions, a severity/risk assessment, recommended specialists, and a downloadable PDF report — with no rigid dropdown menus. Say what's wrong in your own words, in English, Hindi, or Kannada, and the system detects the language, extracts the underlying symptoms, and runs them through a trained classifier.

---

## ✨ Features

- 🌐 **Multilingual input** — detects the input language and auto-translates non-English text before processing (Hindi/Kannada tested)
- 🧠 **Natural-language symptom extraction** — matches both exact dataset terms and everyday phrasing ("can't breathe" → `breathlessness`) via an alias map + spaCy tokenization fallback
- 🔬 **Disease prediction** — a scikit-learn classifier trained on a labeled symptom-disease dataset
- 🚦 **Severity triage** — Mild / Moderate / Urgent, based on per-symptom severity weights, plus a hard **risk flag** for symptoms commonly associated with medical emergencies (e.g. chest pain, breathlessness) so a single serious symptom isn't diluted by an average
- 👨‍⚕️ **Specialist routing** — maps each predicted condition to the relevant type of doctor
- 📊 **Confidence chart & specialist cards** — visual comparison of the top candidate conditions, not just a text list
- 📄 **PDF report export** — generates a shareable report with patient context, detected symptoms, severity, and precautions
- 🗃️ **Query logging** — anonymized symptom/prediction/severity log in SQLite
- 🔌 **Optional REST API** — the same prediction pipeline exposed over FastAPI, independent of the Streamlit UI

---

## 📱 Screenshots

| Symptom Checker Interface | Prediction & Confidence Analysis |
| --------------------------- | ----------------------------------- |
| ![MediCheck symptom input screen — multilingual English, Hindi, Kannada support](https://github.com/saniasagheer05/Medicheck/blob/fb5355253ffda493b07ef82cc07c21e2ec2c4a30/Screenshot%20(233).png) | ![Predicted conditions with confidence scores and comparison chart](https://github.com/saniasagheer05/Medicheck/blob/fb5355253ffda493b07ef82cc07c21e2ec2c4a30/Screenshot%20(236).png) |

| Severity Assessment & Specialist Recommendation | Model Comparison Visualization |
| -------------------------------------------------- | --------------------------------- |
| ![Severity classification and specialist recommendation](https://github.com/saniasagheer05/Medicheck/blob/fb5355253ffda493b07ef82cc07c21e2ec2c4a30/Screenshot%20(237).png) | ![Model comparison across Random Forest, Naive Bayes, and SVM](https://github.com/saniasagheer05/Medicheck/blob/fb5355253ffda493b07ef82cc07c21e2ec2c4a30/Screenshot%20(238).png) |

> User enters symptoms in English, Hindi, or Kannada → the system detects the language, extracts symptoms, predicts conditions with confidence scores, classifies severity (Mild / Moderate / Urgent), and recommends the relevant specialist.

---

## 🏗️ How It Works

```mermaid
flowchart TD
    A[User Input<br/>English / Hindi / Kannada] --> B[symptom_extractor.py]
    B --> C{Language Detection}
    C -->|Non-English| D[Auto-Translate]
    C -->|English| E[Alias Matching +<br/>spaCy Tokenization]
    D --> E
    E --> F[Canonical Symptom<br/>Feature Vector]
    F --> G[predict.py]
    G --> H[Trained Classifier<br/>Random Forest]
    H --> I[Top-N Diseases]
    H --> J[Severity Triage<br/>Mild / Moderate / Urgent]
    H --> K[Specialist Routing]
    I & J & K --> L{Interface}
    L --> M[Streamlit UI<br/>Chart + Cards + PDF]
    L --> N[FastAPI REST<br/>Endpoints]

    style A fill:#e3f2fd,stroke:#1565c0
    style H fill:#fff3e0,stroke:#e65100
    style L fill:#f3e5f5,stroke:#6a1b9a
```

All three pipeline components (`train.py`, `predict.py`, `symptom_extractor.py`) share a single symptom-normalization function in `model/utils.py`, so the feature format used for training always matches the format used at prediction time. See [CHANGELOG.md](CHANGELOG.md) for why this mattered.

### Training Pipeline
![Training Pipeline](https://github.com/saniasagheer05/Medicheck/blob/bbbfc5a348192a01bb01b71c242a6cdc575a2b2f/medicheck_training_pipeline.png)

### Inference Pipeline
![Inference Pipeline](https://github.com/saniasagheer05/Medicheck/blob/bbbfc5a348192a01bb01b71c242a6cdc575a2b2f/medicheck_inference_pipeline.png)

---

## 🔬 Model

Trained on a public Kaggle disease–symptom dataset (4,920 rows, 41 diseases, 131 unique symptoms across up to 17 symptom slots per row). `train.py` converts those slots into a proper binary (multi-hot) feature matrix — one column per symptom — and compares three classifiers:

| Model | Test Accuracy |
|---|---|
| Random Forest | 100.0% |
| Naive Bayes | 100.0% |
| SVM | 100.0% |

The best-performing model (Random Forest, selected automatically by `train.py`) is saved for deployment.

> **Note:** this dataset is small, synthetic, and cleanly-separated, which is why accuracy is at ceiling for all three models — on messier real-world clinical data you'd expect more separation between them, and cross-validation rather than a single train/test split. That's noted here deliberately, since being able to explain *why* a result looks unrealistically perfect is exactly what an interviewer will probe for.

---

## 📂 Project Structure

```
medicheck/
├── app/streamlit_app.py       # Streamlit UI — single entry point
├── api/main.py                # Optional FastAPI REST endpoints
├── model/
│   ├── train.py                # Builds the binary feature matrix & trains models
│   ├── predict.py               # Loads artifacts, runs predictions + severity
│   ├── symptom_extractor.py     # NLP: translation, alias matching, extraction
│   ├── pdf_report.py            # PDF report generation
│   ├── utils.py                 # Shared symptom normalization (single source of truth)
│   ├── medicheck_model.pkl      # Trained classifier (generated by train.py)
│   ├── label_encoder.pkl        # Disease label encoder (generated)
│   ├── symptom_list.pkl         # Canonical symptom feature list (generated)
│   └── model_results.json       # Model comparison results (generated)
├── data/                       # Source CSVs (dataset, severity, description, precautions)
├── tests/                      # pytest suite
├── requirements.txt
└── .streamlit/config.toml      # Theme
```

---

## 🚀 Setup & Run Locally

```bash
git clone https://github.com/saniasagheer05/Medicheck.git
cd Medicheck
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Re)train the model — writes medicheck_model.pkl, label_encoder.pkl,
# symptom_list.pkl, model_results.json into model/. Not required if
# these files are already committed, but useful after changing the data.
python model/train.py

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

Run the test suite:

```bash
pytest tests/ -v
```

Run the optional API instead of/alongside the UI:

```bash
uvicorn api.main:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repo to GitHub (model `.pkl` files must be committed — Streamlit Cloud does not run a training step, it just installs `requirements.txt` and runs the app)
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and select this repo
3. Set **Main file path** to `app/streamlit_app.py`
4. Deploy — first boot may take a minute while spaCy's model wheel installs

---

## ⚠️ Limitations

- Symptom extraction is alias/keyword-based, not a medical NLP model — it won't catch every possible phrasing
- Trained on a synthetic dataset (see [Model](#-model) above) — not validated against real clinical data

---

## 📄 License

For educational/portfolio use.

---

<div align="center">

Built by **Sania Sagheer**

</div>
