import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)


class FisherClassifier():
    def __init__(self):
        self.w = None

    def train(self, X, y):
        '''
        X:(row, col)
        y:(row,)
        return: None
        '''
        row, col = X.shape
        pos = X[y == 1]
        neg = X[y == 0]

        # cal mean
        m1 = np.mean(pos, axis=0)
        m2 = np.mean(neg, axis=0)

        # cal S_w
        sw = (pos - m1).T @ (pos - m1) / pos.shape[0] + (neg - m2).T @ (neg - m2)/neg.shape[0]

        # cal w
        self.w = np.linalg.inv(sw) @ (m1 - m2)

        # store variable for prediction
        self.mu1 = m1 @ self.w
        self.mu2 = m2 @ self.w

        self.s1 = np.mean((pos @ self.w - self.mu1)**2)
        self.s2 = np.mean( (neg @ self.w - self.mu2)**2 )

    def cal_gaussian(self, X, sigma, mu):
        return (1 / (sigma*(2*np.pi)**0.5) ) * np.exp( -(X - mu)**2 / (2*sigma**2) )

    def predict(self, X):
        '''
        X:(row, col)
        return: y: (row, )
        '''
        tmp = X @ self.w
        return int(self.cal_gaussian(tmp, self.s1, self.mu1) > self.cal_gaussian(tmp, self.s2, self.mu2))


if __name__ == '__main__':
    clf = FisherClassifier()

    # data preparation
    m1 = np.array([1, 3])
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
    clf = FisherClassifier()
    clf.train(X, y)
    print(clf.predict(X_test))


    # visualization
    x = np.linspace(-4, 4, 100)
    print(clf.w)
    y = x * (clf.w[1]/clf.w[0]) # for W
    # y = x * (clf.w[0] / clf.w[1]) # for X where xw = 0
    plt.scatter(x, y, c = "g", s = 5)
    plt.scatter(new_data1[:, 0], new_data1[:, 1], c="y")
    plt.scatter(new_data2[:, 0], new_data2[:, 1], c="b")
    plt.scatter(X_test[0, 0], X_test[0, 1], s=50, c="r")
    plt.show()