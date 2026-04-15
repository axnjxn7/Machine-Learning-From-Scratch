import numpy as np
import pandas as pd

class Logistic_Regression:
    def __init__(self, learning_rate = 0.05, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = design_matrix(X)
        self.theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):
            p = self.sigmoid(X @ self.theta)
            error = p - y
            gradient = X.T @ error
            self.theta -= self.learning_rate * gradient

    def pred(self, X):
        X = design_matrix(X)
        p = self.sigmoid(X @ self.theta)
        return (p >= 0.5).astype(int)

def design_matrix(X):
    column_ones = np.ones((X.shape[0], 1))
    return np.hstack((column_ones, X))


if __name__ == '__main__':
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = (X > 1).astype(int).flatten()

    model = Logistic_Regression()
    model.fit(X, y)

    print(model.pred(X[:5]))
    print(y[:5])
