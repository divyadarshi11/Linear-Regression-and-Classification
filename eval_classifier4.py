#DIVYA DARSHI-1002090905
#CSE-6363
#Variant- Petal Length/Width

from matplotlib import pyplot as plt
from mlxtend.evaluate import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis

iris = load_iris()
X = iris.data[:, 2:4] 
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the LDA model for pepal length/width
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LDA (Petal length/width) Accuracy:", accuracy)

# Visualize the decision boundary
plot_decision_regions(X_train, y_train, clf=model)
plt.title('LDA - Petal Length/Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# Save the plot to a file
plt.savefig("eval_classifier4_lda.png")
plt.show()