import numpy as np

class Logistic_Regression:
    def __init__(self, learning_rate = 0.05, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
    
    def sigmoid(self):
        return 1 / (1 + np.exp(-self.theta))

    def fit(self, X, y):
        X = design_matrix(X)
        self.theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        n = len(X)
        for _ in range(self.epochs):
            pass

    def pred(self):
        pass

def design_matrix(X):
    column_ones = np.ones((X.shape[0], 1))
    return np.hstack(column_ones, X)


if __name__ == '__main__':
    X = 2 * np.random.rand()
    y = 

    model = Logistic_Regression()
    model.fit(X, y)
