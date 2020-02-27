import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class LinearRegression():
    def __init__(self, reg = 0.01):
        self.w = None
        self.reg = reg

    def add_b(self, X):
        return np.concatenate([X,np.ones(X.shape[0])[:, None] ], axis=1)

    def train(self, X, y):
        X = self.add_b(X)
        self.w = np.linalg.inv( (X.T @ X + self.reg)) @ X.T @ y

    def predict(self, X):
        X = self.add_b(X)
        return X @ self.w

if __name__ == '__main__':
    # data preparation
    m1 = np.array([-3, 3])
    m2 = np.array([3, 0])
    C1 = np.array([[1, 0], [0, 1]], np.float32)
    C2 = np.array([[0.5, 0], [0, 0.5]], np.float32)
    N = 200
    data1 = np.random.randn(N, 2)
    data2 = np.random.randn(N, 2)
    A1 = np.linalg.cholesky(C1)
    A2 = np.linalg.cholesky(C2)
    new_data1 = data1 @ A1.T + m1
    new_data2 = data2 @ A2.T + m2
    X = np.concatenate([new_data1, new_data2], axis=0)
    print(X.shape)

    y = np.concatenate([np.ones(200), np.zeros(200)])
    print(y.shape)


    # train model
    clf = LinearRegression()
    clf.train(X, y)

    # visualization
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cm = plt.cm.RdBu
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.show()