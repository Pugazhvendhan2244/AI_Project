# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your energy consumption dataset (assuming it's in CSV format)
data = pd.read_csv('energy_consumption_data.csv')

# Data preprocessing: Assume your dataset has columns 'Date', 'EnergyConsumption', and 'OtherFeatures'
# You might need to parse dates and handle missing values, categorical data, etc.

# Extract features and target variable
X = data[['OtherFeatures']]  # Other features affecting energy consumption
y = data['EnergyConsumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model (you can use more complex models as needed)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the predictions
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Energy Consumption Prediction')
plt.xlabel('Other Features')
plt.ylabel('Energy Consumption')
plt.show()
