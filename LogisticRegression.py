#DIVYA DARSHI-1002090905
#CSE-6363

import numpy as np

class LogisticRegression:
    def __init__(self):
        self.theta = None  # Parameters for the logistic regression model

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Add a bias term to the input data
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize parameters
        self.theta = np.zeros(X_bias.shape[1])

        # Use normal equations to find optimal parameters
        self.theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        # Add a bias term to the input data
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Calculate the predicted probabilities
        probabilities = self.sigmoid(X_bias @ self.theta)

        # Threshold the probabilities to make binary predictions
        predictions = (probabilities >= 0.5).astype(int)

        return predictions
