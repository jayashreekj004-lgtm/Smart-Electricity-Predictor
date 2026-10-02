# Smart Electricity Consumption Predictor

Smart Electricity Consumption Predictor is a Machine Learning project that predicts household electricity consumption based on environmental and usage-related factors.

## Project Objective

The objective of this project is to use historical electricity consumption data and Machine Learning to predict future electricity usage.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Linear Regression

## Input Features

The model uses:

- Temperature
- Humidity
- People at Home
- Appliance Usage Hours
- Previous Day Electricity Usage

## Target

The target variable is:

- Electricity Usage (kWh)

## Machine Learning Process

1. Load the electricity dataset
2. Explore and validate the data
3. Visualize electricity consumption patterns
4. Analyze feature relationships
5. Select features and target
6. Split data into training and testing sets
7. Train a Linear Regression model
8. Predict electricity consumption
9. Compare actual and predicted values
10. Evaluate model performance
11. Predict electricity usage for new input data
12. Save the trained model

## Visualizations

The project generates:

- Daily Electricity Consumption
- Temperature vs Electricity Consumption
- Actual vs Predicted Electricity Usage

## Model

Linear Regression is used to predict electricity consumption.

The trained model is saved as:

`electricity_model.pkl`

## Example Prediction

For a new input:

- Temperature: 34°C
- Humidity: 60%
- People at Home: 4
- Appliance Usage: 7.5 hours
- Previous Day Usage: 15.0 kWh

The model predicted approximately:

**15.42 kWh**

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt