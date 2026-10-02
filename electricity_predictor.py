import pandas as pd
import joblib

# Load electricity dataset
data = pd.read_csv("data/electricity_data.csv")

# Display dataset
print("\n--- Electricity Consumption Data ---")
print(data)

# Display basic information
print("\n--- Dataset Information ---")
data.info()

# Display statistical summary
print("\n--- Statistical Summary ---")
print(data.describe())
import matplotlib.pyplot as plt

# Electricity usage trend
plt.figure(figsize=(10, 5))

plt.plot(
    data["Date"],
    data["Electricity_Usage_kWh"],
    marker="o"
)

plt.title("Daily Electricity Consumption")
plt.xlabel("Date")
plt.ylabel("Electricity Usage (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("charts/daily_electricity_consumption.png")
plt.close()

# Temperature vs Electricity Usage
plt.figure(figsize=(8, 5))

plt.scatter(
    data["Temperature"],
    data["Electricity_Usage_kWh"]
)

plt.title("Temperature vs Electricity Consumption")
plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Usage (kWh)")
plt.tight_layout()

plt.savefig("charts/temperature_vs_electricity.png")
plt.close()
# Temperature and Electricity Usage correlation
correlation = data["Temperature"].corr(
    data["Electricity_Usage_kWh"]
)

print("\n--- Temperature vs Electricity Usage Correlation ---")
print("Correlation:", round(correlation, 2))
# Select features and target

features = [
    "Temperature",
    "Humidity",
    "People_Home",
    "Appliance_Usage_Hours",
    "Previous_Day_Usage_kWh"
]

X = data[features]
y = data["Electricity_Usage_kWh"]

print("\n--- Features ---")
print(X.head())

print("\n--- Target ---")
print(y.head())
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Train Test Split ---")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
# Predict electricity consumption for test data
y_pred = model.predict(X_test)
# Compare actual and predicted values

comparison = pd.DataFrame({
    "Actual_Usage": y_test.values,
    "Predicted_Usage": y_pred
})

print("\n--- Actual vs Predicted Electricity Usage ---")
print(comparison)

print("\n--- Predicted Electricity Usage ---")
print(y_pred)

print("\n--- Model Training Completed ---")
print("Linear Regression model trained successfully.")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))
# Actual vs Predicted graph

plt.figure(figsize=(8, 5))

plt.plot(
    y_test.values,
    marker="o",
    label="Actual"
)

plt.plot(
    y_pred,
    marker="o",
    label="Predicted"
)

plt.title("Actual vs Predicted Electricity Usage")
plt.xlabel("Test Data")
plt.ylabel("Electricity Usage (kWh)")
plt.legend()
plt.tight_layout()

plt.savefig("charts/actual_vs_predicted.png")
plt.close()
# Predict electricity usage for a new input

new_data = pd.DataFrame({
    "Temperature": [34],
    "Humidity": [60],
    "People_Home": [4],
    "Appliance_Usage_Hours": [7.5],
    "Previous_Day_Usage_kWh": [15.0]
})

new_prediction = model.predict(new_data)

print("\n--- New Electricity Usage Prediction ---")
print("Predicted Electricity Usage:",
      round(new_prediction[0], 2), "kWh")
# Display feature coefficients

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\n--- Feature Coefficients ---")
print(coefficients)
# Save the trained model

joblib.dump(model, "electricity_model.pkl")

print("\n--- Model Saved ---")
print("Model saved as electricity_model.pkl")