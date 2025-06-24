
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated dataset
data = {
    'Crop': ['Rice', 'Wheat', 'Maize', 'Rice', 'Wheat', 'Maize'],
    'Soil_Type': [0, 1, 2, 0, 1, 2],
    'Fertilizer': ['Urea', 'DAP', 'NPK', 'Urea', 'DAP', 'NPK']
}

df = pd.DataFrame(data)

# Features and target variable
X = df.drop(['Fertilizer'], axis=1)
y = df['Fertilizer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Testing and evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Fertilizer Recommendation Model Accuracy: {accuracy * 100:.2f}%")
