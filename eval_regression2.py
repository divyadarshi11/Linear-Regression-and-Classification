#DIVYA DARSHI-1002090905
#CSE-6363
#MODEL-2

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression  

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Define the input and output features 
input_features = [1, 2]  # Sepal width and Petal width
target_features = [0]    # Sepal length

# Normalize the input features 
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_test_normalized = (X_test - X_train_mean) / X_train_std

# Create an instance of  LinearRegression class 
regression_model2 = LinearRegression(input_features, target_features)

# Load the trained model parameters from the file
regression_model2.weights = np.load('model2_parameters.npy')

# Make predictions on the test data
y_test_pred = regression_model2.predict(X_test_normalized)

# Calculate the Mean Squared Error for test data
mse_test = np.mean((y_test_pred - y_test) ** 2)

# Print the MSE for the test data
print(f'Test Mean Squared Error for model2: {mse_test:.4f}')
