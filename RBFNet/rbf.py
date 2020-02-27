import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap


class RBFNet():
    def __init__(self, M, r=0.1, reg=0.001):
        '''
        m: 升维度后的维度
        r: rbf 的参数
        reg: 正则化参数
        '''
        self.M = M
        self.r = r
        self.reg = reg

        # setting for logistic regression
        self.lr = 0.01
        self.epoch = 100
        self.w = None
        self.b = None

    def rbf(self, x, u):
        return np.exp( - ((x - u) @ (x-u)) / self.r)

    def __cal_center(self, X):
        clf = KMeans(n_clusters=self.M)
        clf.fit(X)
        return clf.cluster_centers_

    def __sigmoid(self, z):
        return 1 / (1 + np.e ** (-z))

    def __linear_regression_train(self, G):
        self.w = np.linalg.inv(G.T @ G + self.reg*np.diag(G.shape[1])) @ G.T @ y

    def __linear_regression_predict(self, G):
        return G @ self.w

    def __logistic_regression_train(self, X, y):
        # initialize W and b
        N, D = X.shape
        y = y.reshape((N, 1))
        self.w = np.random.standard_normal(size=(D, 1))
        self.b = np.random.standard_normal(size=(1, 1))

        for i in range(self.epoch):
            # forward
            z = X.dot(self.w)
            a = self.__sigmoid(z)
            # loss = -1.0 / N * (y.T.dot(np.log(a)) + (1 - y).T.dot(np.log(1 - a)))

            # backward
            dw = X.T.dot(a - y) / N
            db = np.mean(a - y)

            # update "
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def __logistic_regression_predict(self, X, thresold = 0.5):
        z = X.dot(self.w)
        a = self.__sigmoid(z)
        # return (a > thresold).astype(int)
        return a

    def __cal_G(self, X):
        N, p = X.shape
        G = np.zeros((N, self.M))

        for i in range(N):
            for j in range(self.M):
                G[i, j] = self.rbf(X[i, :], self.C[j, :])
        return G


    def train(self, X, y):

        self.C = self.__cal_center(X)

        G = self.__cal_G(X)

        self.__logistic_regression_train(G, y)

    def predict(self, X):
        G = self.__cal_G(X)
        return self.__logistic_regression_predict(G)


if __name__ == '__main__':
    X, y = make_moons(noise=0.3, random_state=0)
    clf = RBFNet(M=10)
    clf.train(X, y)

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