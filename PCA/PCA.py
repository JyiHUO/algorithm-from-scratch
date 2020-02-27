import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, n):
        '''
        n: dimention reduce
        '''
        self.w = None # (col, n)
        self.n = n

    def train(self, X):
        '''
        X: (row, col)
        '''

        new_x = (X - np.mean(X, 0)) / X.shape[0]**0.5

        U, sigma, VT = np.linalg.svd(new_x)

        self.w = VT.T[:, :self.n]

    def predict(self, X):
        return X @ self.w


if __name__ == '__main__':

    # data preparation
    m1 = np.array([-2, 3])
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

    X_test = np.array([[-2, 7]])

    # train model
    clf = PCA(n=1)
    clf.train(X)
    new_x = clf.predict(X=X)


    # visualization
    plt.scatter(X[:, 0], X[:, 1], c="y")
    x = np.linspace(-4, 4, 100)
    print(clf.w)
    y = x * (clf.w[1] / clf.w[0])  # for W
    plt.scatter(x, y, c="g", s=5)
    plt.show()