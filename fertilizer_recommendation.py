import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Get current directory
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fertilizer_model.pkl")

# Simulated dataset (replace with real data if available)
data = {
    'Crop': ['Rice', 'Wheat', 'Maize', 'Rice', 'Wheat', 'Maize'],
    'Soil_Type': [0, 1, 2, 0, 1, 2],
    'Fertilizer': ['Urea', 'DAP', 'NPK', 'Urea', 'DAP', 'NPK']
}

df = pd.DataFrame(data)

# Features and target
X = df.drop(['Fertilizer'], axis=1)
y = df['Fertilizer']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🧪 Fertilizer Recommendation Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Fertilizer model saved as '{MODEL_PATH}'")


