import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load dataset and model
data = pd.read_csv("data/dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]

model = joblib.load("models/model.pkl")

# Evaluate model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy}")
