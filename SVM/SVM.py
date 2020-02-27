import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap


class SVM():
    def __init__(self, C, epoch = 10):
        self.C = C
        self.w = None
        self.b = None
        self.alpha = None
        self.X = None
        self.y = None
        self.epoch = epoch

    def init(self, X):
        row, col = X.shape
        self.w = [np.random.normal() for c in range(col)]
        self.b = np.random.normal()
        self.alpha = [np.random.normal() for n in range(row)]

    def f(self, x):
        return x @ self.w + self.b

    def K(self, x1, x2):
        return np.exp( - (x1 - x2) @ (x1-x2) / 0.5)

    def select(self, X, i):
        n = np.random.randint(X.shape[0])
        while n == i:
            n = np.random.randint(X.shape[0])
        return n

    def train(self, X, y):
        self.init(X)
        self.X = X
        self.y = y

        row, col = X.shape
        for e in range(self.epoch):
            for i in range(row):
                id1 = i
                id2 = self.select(X, i)
                x1 = X[i, :]
                x2 = X[id2, :]
                y1 = y[i]
                y2 = y[id2]

                E1 = self.f(x1) - y1
                E2 = self.f(x2) - y2

                eta = self.K(x1, x1) + self.K(x2, x2) - 2*self.K(x1, x2)

                alpha2_clipped = self.alpha[id2] + y2*(E1 - E2)/eta

                if y1 != y2:
                    L = max(0, self.alpha[id2] - self.alpha[id1])
                    H = min(self.C, self.alpha[id2] - self.alpha[id1] + self.C)
                else:
                    L = max(0, self.alpha[id1] + self.alpha[id2] - self.C)
                    H = min(self.C, self.alpha[id1] + self.alpha[id2])

                if H < alpha2_clipped:
                    alpha2 = H
                elif L <= alpha2_clipped <= H:
                    alpha2 = alpha2_clipped
                else:
                    alpha2 = L

                alpha1 = self.alpha[id1] + y1*y2*(self.alpha[id2] - alpha2)

                b1 = y1 - np.sum([ self.alpha[n]*y[n]*self.K(X[n,:], x1) for n in range(row) if not (n in {id1, id2}) ]) \
                     - alpha1 * y1* self.K(x1, x1) - alpha2*y2*self.K(x2, x1)
                b2 = -E2 - y1 * self.K(x1, x2) * (alpha1 - self.alpha[id1]) - y2 * self.K(x2,x2) * (alpha2 - self.alpha[id2]) + self.b
                b = (b1 + b2) / 2

                # update
                self.b = b
                self.alpha[id1] = alpha1
                self.alpha[id2] = alpha2
                self.w = ((np.array(self.alpha) * y) @ X).flatten()

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            tmp = 0
            for j in range(self.X.shape[0]):
                if self.alpha[j] == 0:
                    continue
                tmp += self.alpha[j] * self.y[j] * self.K(self.X[j,:], X[i, :])
            y.append(tmp + self.b)
        return np.array(y)


if __name__ == '__main__':
    X, y = make_moons(noise=0.3, random_state=0, n_samples=50)
    y[y==0] = -1
    clf = SVM(C = 1)
    clf.train(X, y)
    print(clf.alpha)

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



