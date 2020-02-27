import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron():
    def __init__(self, alpha = 0.01, epoch=20):
        self.alpha = alpha
        self.epoch = epoch
        self.w = None
        self.b = None

    def init(self, X):
        self.w = np.array([np.random.normal() for i in range(X.shape[1])])
        self.b = np.random.normal()

    def train(self, X, y):
        self.init(X)

        for e in range(self.epoch):
            for i in range(X.shape[0]):
                if y[i] * (X[i, :] @ self.w + self.b) < 0:
                    self.w = self.w + self.alpha * y[i] * X[i, :]
                    self.b = self.b + self.alpha*y[i]

    def predict(self, X):
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

    y = np.concatenate([np.ones(200), -1*np.ones(200)])
    print(y.shape)

    # train model
    clf = Perceptron()
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