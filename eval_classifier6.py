#DIVYA DARSHI-1002090905
#CSE-6363
#Variant- All Features

from matplotlib import pyplot as plt
from mlxtend.evaluate import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis

iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the LDA model using all features
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LinearDiscriminantAnalysis (All Features) Accuracy:", accuracy)
