import pandas as pd
import joblib

df = pd.read_csv("data/dataset.csv")

print("Columns:", df.columns)

symptoms = []

for col in df.columns[1:]:
    cleaned = df[col].dropna().apply(lambda x: x.strip().replace("_", " "))
    symptoms.extend(cleaned.unique())

symptoms = list(set(symptoms))

print("Sample symptoms:", symptoms[:10])  

joblib.dump(symptoms, "model/symptom_list.pkl")

print("symptom_list.pkl created successfully!")