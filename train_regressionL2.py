#DIVYA DARSHI-1002090905
#CSE-6363

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression  

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Define the input and target features 
input_features = [0, 1]  #Sepal length and Sepal width
target_features = [3]    #Petal width

# Create an instance of LinearRegression class without regularization
regression_model = LinearRegression(input_features, target_features)

# Train the model without regularization using batch gradient descent
losses = regression_model.fit(X_train, y_train, regularization=0.0)

# Calculate the weights of the non-regularized model
weights_non_regularized = regression_model.weights

# Create an instance of LinearRegression class with L2 regularization
regression_model_regularized = LinearRegression(input_features, target_features)

# Train the model with L2 regularization using batch gradient descent
losses_regularized = regression_model_regularized.fit(X_train, y_train, regularization=0.1)

# Calculate the weights of the regularized model
weights_regularized = regression_model_regularized.weights

# Calculate the difference in weights between regularized and non-regularized models
weight_difference = weights_non_regularized - weights_regularized

# Print the weight difference
print("Weight Difference (Non-Regularized - Regularized):\n", weight_difference)

# Plot the loss against the step number 
plt.plot(range(len(losses)), losses)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training with L2 - Model 1')
plt.savefig('loss_plotL2.png') 
plt.show()
