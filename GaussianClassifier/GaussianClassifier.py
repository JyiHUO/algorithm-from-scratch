import numpy as np
np.random.seed(10)


class GaussianClassifier():
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.pos_mu = None
        self.pos_C = None
        self.neg_mu = None
        self.neg_C = None

    def cal_gaussian(self, X, C, mu):
        '''
        :param X: (batch_size, col)
        :param C: covariance matrix (col, col)
        :param mu: mean vector (col, )
        :return: (batch_size, )
        '''
        k = X.shape[1]
        return np.exp(-0.5*  (((X - mu) @ np.linalg.inv(C)) * (X - mu)).sum() ) / np.sqrt( (2 * np.pi)**k * np.linalg.det(C) )

    def cal_C(self, X):
        '''
        :param X: (batch_size, col)
        :return: (col, col)
        '''
        return np.cov(X, rowvar=False)

    def cal_mu(self, X):
        '''
        :param X: (batch_size, col)
        :return: (col,)
        '''
        return np.mean(X, axis=0)

    def train(self, X, y):
        '''
        :param X: (batch_size, col)
        :param y: (batch_size,)
        :return:
        '''
        pos_X = X[y == 1]
        neg_X = X[y == 0]
        self.pos_mu = self.cal_mu(pos_X)
        self.pos_C = self.cal_C(pos_X)
        self.neg_mu = self.cal_mu(neg_X)
        self.neg_C = self.cal_C(neg_X)

    def predict(self, X_test):
        '''
        :param X_test: (batch_size, col)
        :return: (batch_size, )
        '''
        g1 = self.cal_gaussian(X_test, self.pos_C, self.pos_mu)
        g2 = self.cal_gaussian(X_test, self.neg_C, self.neg_mu)
        return int(1 / (1 + self.alpha * (g2 / g1)) > 0.5)


if __name__ == '__main__':
    # data preparation
    m1 = np.array([0, 3])
    m2 = np.array([3, 2.5])
    C1 = np.array([[2, 0], [0, 2]], np.float32)
    C2 = np.array([[1.5, 0], [0, 1.5]], np.float32)


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

    X_test = np.array([[2, -2]])

    # train model
    clf = GaussianClassifier()
    clf.train(X, y)
    print(clf.predict(X_test))