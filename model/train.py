import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip().lower().replace(" ", "_") if isinstance(x, str) else x)

X = df.drop("Disease", axis=1)
y = df["Disease"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
symptom_list = list(X.columns)
print(f"Loaded {len(df)} rows, {len(symptom_list)} symptoms, {len(le.classes_)} diseases")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = round(acc * 100, 2)
    print(f"{name} Accuracy: {results[name]}%")

best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"Best model: {best_name} ({results[best_name]}%)")

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/medicheck_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(symptom_list, "model/symptom_list.pkl")

with open("model/model_results.json", "w") as f:
    json.dump(results, f)

print("Model saved successfully!")