import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

y_prob = log_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("images/roc_curve_logistic.png", dpi=300, bbox_inches="tight")
plt.show()