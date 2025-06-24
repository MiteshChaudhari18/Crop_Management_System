import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulated dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020],
    'Rainfall': [200, 220, 250, 230, 240, 260],
    'Yield': [2.5, 2.7, 3.0, 2.8, 3.1, 3.2]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['Year', 'Rainfall']]
y = df['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Testing and evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Rainfall and Yield Prediction")
print(f"Predictions: {predictions}")
print(f"Mean Squared Error: {mse:.2f}")
