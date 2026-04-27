import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("covid_data.csv")

df.replace([97, 98, 99], np.nan, inplace=True)

df.dropna(inplace=True)

df["target"] = df["date_died"].apply(lambda x: 0 if x == "9999-99-99" else 1)

features = [
    "age", "sex", "pneumonia", "diabetes", "hypertension",
    "cardiovascular", "renal_chronic", "copd", "asthma",
    "obesity", "tobacco", "inmsupr", "other_disease",
    "patient_type", "intubed", "icu"
]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)
