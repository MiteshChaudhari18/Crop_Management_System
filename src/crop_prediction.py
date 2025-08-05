import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # for saving the trained model

# Load dataset
data = pd.read_csv("crop_data.csv")

# Features and target variable
X = data.drop(['Crop'], axis=1)
y = data['Crop']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Crop Prediction Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(model, "crop_model.pkl")
print("✅ Model saved as 'crop_model.pkl'")

