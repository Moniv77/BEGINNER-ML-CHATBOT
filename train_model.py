import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("data.csv")

# Encode text to numbers
encoder = LabelEncoder()
data["skin_type"] = encoder.fit_transform(data["skin_type"])
data["skin_tone"] = encoder.fit_transform(data["skin_tone"])
data["concern"] = encoder.fit_transform(data["concern"])

X = data[["skin_type", "skin_tone", "concern"]]
y = data["recommendation"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")

