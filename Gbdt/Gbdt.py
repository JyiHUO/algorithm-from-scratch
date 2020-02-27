import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class GBDT():
    def __init__(self, n_estimates=10, regression=False, clf=DecisionTreeRegressor):
        self.n_estimates = n_estimates
        self.regression = regression
        self.estimates = [clf(max_depth=2) for i in range(n_estimates)]

    def cross_entropy(self, y_pred, y_true):
        return np.sum(- np.log(y_pred) * y_true)

    def softmax(self, y_pred):
        return np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

    def gradient(self, y_pred, y_true):
        return y_true - y_pred

    def fit(self, X, y):
        '''
        y:(n_sample, K)
        '''
        self.estimates[0].fit(X, y)
        y_pred = self.estimates[0].predict(X)
        for i in range(1, self.n_estimates):
            print("loss: ", self.cross_entropy(self.softmax(y_pred), y))
            self.estimates[i].fit(X, y - y_pred)
            y_pred +=  self.estimates[i].predict(X)

    def predict(self, X):
        y_pred = 0
        for i in range(self.n_estimates):
            y_pred += self.estimates[i].predict(X)
        if not self.regression:
            return np.argmax(self.softmax(y_pred), axis=1)
        else:
            return y_pred


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    y = np.stack([(y==0).astype(int), (y==1).astype(int)]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = GBDT()
    feature_type = [0 for i in range(X.shape[1])]
    clf.fit(X_train, y_train)
    # print(y_test)
    print(accuracy_score(np.argmax(y_test, axis=1), clf.predict(X_test) ))