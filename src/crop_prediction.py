
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("crop_data.csv")

# Features and target variable
X = data.drop(['Crop'], axis=1)
y = data['Crop']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Testing and evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Crop Prediction Model Accuracy: {accuracy * 100:.2f}%")
