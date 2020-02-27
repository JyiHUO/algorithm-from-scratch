import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DecisionTree.decision_tree_classifier import DecisionTreeClassifier


class Adaboost():
    def __init__(self, n_estimators=10, clf=DecisionTreeClassifier):
        self.estimators = [clf(max_depth=2) for i in range(n_estimators)]
        self.n_estimators = n_estimators
        self.alpha = [0 for i in range(n_estimators)]

    def fit(self, X, y, feature_type):
        self.estimators[0].fit(X, y, feature_type)
        G = np.ones_like(y, dtype=np.float)
        for i in range(0, self.n_estimators):
            self.estimators[i].fit(X, y, feature_type)
            y_pred = self.estimators[i].predict(X)
            w_pre = np.exp(- (y*2 - 1) * G) / np.sum( np.exp(- (y*2 - 1) * G) )
            self.alpha[i] = np.log( np.sum(w_pre*(y==y_pred) ) / np.sum(w_pre*(y!=y_pred)) + 10**-5)/2
            G += self.alpha[i] * y_pred

    def predict(self, X):
        Z = np.sum(self.alpha)
        y_pred = 0
        for i in range(self.n_estimators):
            y_pred += self.estimators[i].predict(X) * self.alpha[i] / Z
        return y_pred


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    # y = np.stack([(y==0).astype(int), (y==1).astype(int)]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = Adaboost()
    y_train = y_train[:, None]
    feature_type = [0 for i in range(X.shape[1])]
    clf.fit(X_train, y_train, feature_type=feature_type)
    print(accuracy_score(y_test, clf.predict(X_test) > 0.5))