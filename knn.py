import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("archive/Covid_Data.csv")

df["target"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)

df = df.drop(columns=["DATE_DIED"])

df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

df = df.replace([97, 98, 99], pd.NA)

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)


y_pred = knn_model.predict(X_test_scaled)


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))