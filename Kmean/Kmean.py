import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(2020)

class Kmeans():
    def __init__(self, K, epoch=20):
        self.epoch = epoch
        self.K = K
        self.centers = None

    def __cal_d(self, x, c):
        return (x - c) @ (x - c)

    def train(self, X):
        centers = X[ [np.random.randint(X.shape[0]) for i in range(self.K)] ]

        for i in range(self.epoch):
            centers_map = [[] for i in range(len(centers))]
            for i in range(X.shape[0]):
                min_d = 10 ** 8
                min_id = None
                for j in range(len(centers) ):
                    d = self.__cal_d(X[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_id = j
                centers_map[min_id].append(i)

            # update
            for i in range(len(centers)):
                if len(centers_map[i]) == 0:
                    continue
                else:
                    centers[i] = np.mean( X[centers_map[i]], axis=0 )
        self.centers = centers

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            min_d = 10 ** 8
            min_id = None
            for j in range(len(self.centers) ):
                d = self.__cal_d(X[i], self.centers[j])
                if d < min_d:
                    min_d = d
                    min_id = j
            y.append(min_id)
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
    clf = Kmeans(K=2)
    clf.train(X)
    y = clf.predict(X)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.show()
