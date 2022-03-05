import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/heart_model.pkl")
print("Model saved.")