import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("archive/Covid_Data.csv")

df["target"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)
df = df.drop(columns=["DATE_DIED"])
df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]
df = df.replace([97, 98, 99], pd.NA)
df = df.dropna()

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.colorbar()
plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
plt.yticks([0, 1], ["Actual 0", "Actual 1"])
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title("Confusion matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("confusion_matrix_logistic.png", dpi=300, bbox_inches="tight")
plt.show()