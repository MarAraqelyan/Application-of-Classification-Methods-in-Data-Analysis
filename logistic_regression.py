import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("archive/Covid_Data.csv")

df["target"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)

df = df.drop(columns=["DATE_DIED"])

df = df.replace([97, 98, 99], pd.NA)

df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

df = df.dropna()

X = df.drop(columns=["target"])
y = df["target"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data count:", len(X_train))
print("Test data count:", len(X_test))

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))