import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("vehicle_data.csv")

# Input and output
X = data.drop("failure", axis=1)
y = data["failure"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "vehicle_model.pkl")

print("Model Saved")
