import numpy as np
import matplotlib.pyplot as plt
import os

class Linear_Regression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X = design_matrix(X)
        self.theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        n = len(X)
        self.loss_history = [] 

        for _ in range(self.epochs):
            y_yhat = X @ self.theta
            residuals = y - y_yhat
            loss = (1 / n) * np.sum(residuals ** 2)
            self.loss_history.append(loss)
            gradient = (-2/n) * X.T @ residuals
            self.theta -= self.learning_rate * gradient

    def pred(self, X):
        X = design_matrix(X)
        return X @ self.theta
    

def r_square(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_total)


def design_matrix(X):
    col_ones = np.ones((X.shape[0], 1))
    return np.hstack((col_ones, X))

def plots():
    try:

        folder = "linreg_images"
        os.makedirs(folder, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.plot(model.loss_history)
        plt.savefig(os.path.join(folder, "loss.png"))
        plt.close()

        file_path = os.path.join(folder, "Actual vs Predicted.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, label="Actual")
        plt.plot(X, y_pred, color="Red", label="Predicted")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Predicted vs Actual")
        plt.savefig(os.path.join(folder, "Actual vs Predicted.png"))
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")



if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) # y = b + wx + noise 

    model = Linear_Regression()
    model.fit(X, y)

    y_pred = model.pred(X)
    r2 = r_square(y, y_pred)
    b, w = model.theta.flatten()

    plots()

    print("Actual function: y = 4 + 3x")
    print(f"Predicted function: y = {b:.3f} + {w:.3f}x\nR-Squared: {r2:.3f}")

