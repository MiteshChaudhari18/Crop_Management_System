import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# Set file path for saving the model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "yield_model.pkl")

# Simulated dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Rainfall': [200, 220, 250, 230, 240, 260],
    'Yield': [2.5, 2.7, 3.0, 2.8, 3.1, 3.2]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Year', 'Rainfall']]
y = df['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("📈 Rainfall and Yield Prediction")
print(f"Predictions: {predictions}")
print(f"Mean Squared Error: {mse:.2f}")

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"✅ Yield model saved as '{MODEL_PATH}'")


