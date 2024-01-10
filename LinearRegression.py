#DIVYA DARSHI-1002090905
#CSE-6363

import numpy as np

class LinearRegression:
    def __init__(self, input_features, target_features, learning_rate=0.01, max_epochs=100, batch_size=32, patience=3):
        self.input_features = input_features
        self.target_features = target_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weights = None
        self.losses = []

    def fit(self, X_train, y_train, regularization):
        # Prepare the training data for the model
        X_train_model1 = X_train[:, self.input_features]
        y_train_model1 = X_train[:, self.target_features]

        # Initialize weights
        self.weights = np.random.rand(X_train_model1.shape[1], y_train_model1.shape[1])
        best_loss = float('inf')
        patience_count = 0

        for epoch in range(self.max_epochs):
            # Shuffle the training data
            indices = np.random.permutation(len(X_train_model1))
            X_train_model1_shuffled = X_train_model1[indices]
            y_train_model1_shuffled = y_train_model1[indices]

            # Split the training data into batches
            num_batches = len(X_train_model1) // self.batch_size
            for i in range(num_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_train_model1_shuffled[start:end]
                y_batch = y_train_model1_shuffled[start:end]

                # Perform batch gradient descent
                predictions = np.dot(X_batch, self.weights)
                error = predictions - y_batch
                gradient = np.dot(X_batch.T, error) + regularization * self.weights

                self.weights -= self.learning_rate * gradient / self.batch_size

            # Calculate the mean squared error loss
            predictions = np.dot(X_train_model1, self.weights)
            
            if y_train_model1.shape[1] == 1:  # Single output
                loss = np.mean((predictions - y_train_model1) ** 2)
            else:  # Multiple outputs
                loss = np.mean(np.sum((predictions - y_train_model1) ** 2, axis=1))
            
            self.losses.append(loss)

            # Check for early stopping
            if loss < best_loss:
                best_loss = loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        return self.losses

    def predict(self, X):
        # Extract the relevant input features for prediction
        X_model = X[:, self.input_features]
        # Calculate predictions using the learned weights
        predictions = np.dot(X_model, self.weights)
        return predictions

    def score(self, X, y):
        X_model = X[:, self.input_features]
        y_model = y[:, self.target_features]  # multiple outputs
        predictions = np.dot(X_model, self.weights)
        
        if y_model.shape[1] == 1:  # Single output
            mse = np.mean((predictions - y_model) ** 2)
        else:  # Multiple outputs
            mse = np.mean(np.sum((predictions - y_model) ** 2, axis=1))
        
        return mse
