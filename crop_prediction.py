import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Get the path to the directory this file is in
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "crop_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Features and target
X = data.drop(['Crop'], axis=1)
y = data['Crop']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"🌱 Crop Prediction Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved as '{MODEL_PATH}'")


