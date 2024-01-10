#DIVYA DARSHI-1002090905
#CSE-6363

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import LinearRegression  # Import your custom LinearRegression class

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Select the features for input and target
input_features = [0, 1]  # Sepal length and Sepal width
target_features = [2, 3]  # Petal length and Petal width

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Standardize the input features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Standardize the output features 
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Create an instance of LinearRegression class
regression_model = LinearRegression(input_features, target_features)

# Train the model using batch gradient descent
losses = regression_model.fit(X_train, y_train_scaled, regularization=0.0)

# Save the trained model weights to a NumPy binary file
np.save("linear_regression_weights.npy", regression_model.weights)

# Load the trained model weights from the saved file
loaded_weights = np.load("linear_regression_weights.npy")
loaded_model = LinearRegression(input_features, target_features)
loaded_model.weights = loaded_weights

# Calculate the mean squared error for the training set using the loaded model
y_train_pred_scaled = loaded_model.predict(X_train)
mse_train_scaled = np.mean((y_train_pred_scaled - y_train_scaled) ** 2)

# Calculate the mean squared error for the test set using the loaded model
y_test_pred_scaled = loaded_model.predict(X_test)
mse_test_scaled = np.mean((y_test_pred_scaled - y_test_scaled) ** 2)

# Print the training and test MSE for both output features using the loaded model
print(f'Training Mean Squared Error: {mse_train_scaled:.4f}')
print(f'Test Mean Squared Error: {mse_test_scaled:.4f}')

