#DIVYA DARSHI-1002090905
#CSE-6363

import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.shared_cov_matrix = None
        self.class_means = None
        self.class_priors = None
        self.num_classes = None

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.shared_cov_matrix = np.cov(X, rowvar=False)
        
        self.class_means = []
        self.class_priors = []
        
        for i in range(self.num_classes):
            X_class_i = X[y == i]
            class_mean_i = np.mean(X_class_i, axis=0)
            self.class_means.append(class_mean_i)
            self.class_priors.append(len(X_class_i) / len(X))
        
    def predict(self, X):
        predictions = []
        
        for x in X:
            class_scores = []
            
            for i in range(self.num_classes):
                mean_diff = x - self.class_means[i]
                class_score = -0.5 * mean_diff @ np.linalg.inv(self.shared_cov_matrix) @ mean_diff.T + np.log(self.class_priors[i])
                class_scores.append(class_score)
            
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        
        return np.array(predictions)
