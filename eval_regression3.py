#DIVYA DARSHI-1002090905
#CSE-6363
#MODEL - 3

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

# Load the model parameters saved during training
model3_parameters = np.load('model3_parameters.npy')

# Create an instance of LinearRegression class with the same input and target features
input_features = [1, 2]  
target_features = [3]    
regression_model3 = LinearRegression(input_features, target_features)

# Set the model parameters to the loaded values
regression_model3.weights = model3_parameters
# Predict the target values on the test set
y_pred = regression_model3.predict(X_test)

# Calculate the mean squared error
mse = np.mean((y_pred - y_test) ** 2)

# Print the mean squared error to the console
print(f'Test Mean Squared Error for Model 3: {mse}')

