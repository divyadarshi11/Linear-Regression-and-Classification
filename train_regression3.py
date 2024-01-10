#DIVYA DARSHI-1002090905
#CSE-6363
#MODEL-3

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

# Define the input and target features for model
input_features = [1, 2]  #sepal width and Petal length
target_features = [3]    #petal width

# Create an instance of LinearRegression class
regression_model3 = LinearRegression(input_features, target_features)

# Train the model using batch gradient descent
losses = regression_model3.fit(X_train, y_train, regularization=0.0)

# Save the trained model parameters to a file
np.save('model3_parameters.npy', regression_model3.weights)   

# Plot the loss against the step number 
plt.plot(range(len(losses)), losses)
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Predicting petal width from Sepal width and petal length')
plt.savefig('loss_plot3.png')  
plt.show()




