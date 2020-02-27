import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Utili.visualization import *
np.random.seed(2020)

class MixedGaussian():
    def __init__(self, K, epoch=20):
        self.K = K
        self.mu = None
        self.Sigma = None
        self.pi = None
        self.epoch=epoch

    def init(self, X):
        N = X.shape[0]
        self.mu = X[[np.random.randint(N) for i in range(self.K)]]
        self.Sigma = [ ((X - self.mu[k]).T @ (X - self.mu[k])) / N for k in range(self.K)]
        self.pi = [1/N for i in range(self.K)]

    def cal_gaussian(self, X, C, mu):
        '''
        :param X: (batch_size, col)
        :param C: covariance matrix (col, col)
        :param mu: mean vector (col, )
        :return: (batch_size, )
        '''
        if len(X.shape) == 1:
            X = X[None, :]
        k = X.shape[1]
        return np.exp(-0.5*  (((X - mu) @ np.linalg.inv(C)) * (X - mu)).sum() ) / np.sqrt( (2 * np.pi)**k * np.linalg.det(C) )

    def train(self, X):
        self.init(X)
        N, C = X.shape

        r = np.zeros( (N, self.K) )
        for e in range(self.epoch):
            for n in range(N):
                z = np.sum([self.pi[k] * self.cal_gaussian(X[n, :], self.Sigma[k], self.mu[k]) for k in range(self.K)])
                for k in range(self.K):
                    r[n, k] = self.pi[k]*self.cal_gaussian(X[n, :], self.Sigma[k], self.mu[k]) / z

            # update
            Nk = r.sum(axis=0)  # (k,)
            self.pi = Nk / N  # (k, )
            self.mu = [ (X.T @ r[:, k]) / Nk[k]  for k in range(self.K)]
            self.Sigma =  [ ( ( r[:, k][:, None]* (X - self.mu[k]) ).T @ (X - self.mu[k]) ) /Nk[k] for k in range(self.K) ]

    def predict(self, X):
        y = []
        for x in X:
            y.append( np.argmax( [self.cal_gaussian(x, self.Sigma[k], self.mu[k]) for k in range(self.K)]))
        return y


if __name__ == '__main__':


    # data preparation
    m1 = np.array([1, 3])
    m2 = np.array([3, 0])
    C1 = np.array([[3, 0], [0, 3]], np.float32)
    C2 = np.array([[0.5, 0], [0, 0.5]], np.float32)
    N = 200
    data1 = np.random.randn(N, 2)
    data2 = np.random.randn(N, 2)
    A1 = np.linalg.cholesky(C1)
    A2 = np.linalg.cholesky(C2)
    new_data1 = data1 @ A1.T + m1
    new_data2 = data2 @ A2.T + m2
    X = np.concatenate([new_data1, new_data2], axis=0)

    # model
    clf = MixedGaussian(K=2)
    clf.train(X)
    y = clf.predict(X)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)

    nx, ny = 50, 50
    X1, Y1, Z1 = twoD(nx, ny, m=clf.mu[0], C=clf.Sigma[0])
    X2, Y2, Z2 = twoD(nx, ny, m=clf.mu[1], C=clf.Sigma[1])

    plt.contour(X1, Y1, Z1, 3)
    plt.contour(X2, Y2, Z2, 3)

    plt.show()


