#DIVYA DARSHI-1002090905
#CSE-6363
#MODEL-2

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

# Define the input and output features 
input_features = [1, 2]  # Sepal width and Petal length
target_features = [0]    # sepal length

# Normalize the input features 
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_normalized = (X_train - X_train_mean) / X_train_std

# Create an instance of LinearRegression class with regularization
regression_model2 = LinearRegression(input_features, target_features)

# Train the model using batch gradient descent
losses = regression_model2.fit(X_train_normalized, y_train, regularization=0.0)

# Save the trained model parameters to a file
np.save('model2_parameters.npy', regression_model2.weights)

# Plot the loss against the step number and save the plot
plt.plot(range(len(losses)), losses)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Predicting petal length from Sepal width and sepal length')
plt.savefig('loss_plot2.png') 
plt.show()

