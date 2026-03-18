import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("archive/Covid_Data.csv")

# Create target
df["target"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)

# Drop original column
df = df.drop(columns=["DATE_DIED"])

# Replace missing values
df = df.replace([97, 98, 99], pd.NA)

# Filter only covid positive
df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

# Drop missing
df = df.dropna()

# Split X and y
X = df.drop(columns=["target"])
y = df["target"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Coefficients
coeffs = model.coef_[0]
features = X.columns

df_coeff = pd.DataFrame({
    "Feature": features,
    "Coefficient (β)": coeffs,
    "Odds Ratio": np.exp(coeffs)
})

# Sort
df_coeff = df_coeff.sort_values(by="Coefficient (β)", ascending=False)

print(df_coeff)