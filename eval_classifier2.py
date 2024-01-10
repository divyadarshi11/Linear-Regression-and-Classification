#DIVYA DARSHI-1002090905
#CSE-6363
#Variant- Sepal Length/Width


from matplotlib import pyplot as plt
from mlxtend.evaluate import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = load_iris()
X = iris.data[:, 0:2] 
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Create and train the logistic regression model for petal length/width
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression (Sepal length/width) Accuracy:", accuracy)

# Visualize the decision boundary
plot_decision_regions(X_train, y_train, clf=model)
plt.title('Logistic Regression - Sepal Length/Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
# Save the plot to a file
plt.savefig("eval_classifier2.png")
plt.show()
