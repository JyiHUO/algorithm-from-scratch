import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from DecisionTree.decision_tree import DecisionTree
from DecisionTree.decision_tree_parallel import DecisionTreeParallel
import time

class XGBoostTree(DecisionTreeParallel):
    def __init__(self, lamda=0.001, max_depth=2, min_sample_leaf=4):
        super().__init__(max_depth=max_depth, min_sample_leaf=min_sample_leaf)
        self.lamda = lamda

    def _impurity_calculation(self, y, y1, y2):
        Gl = np.sum(self.__gradient(y1[:, 0], y1[:, 1]))
        Hl = np.sum(self.__hessian(y1[:, 0], y1[:, 1]))
        Gr = np.sum(self.__gradient(y2[:, 0], y2[:, 1]))
        Hr = np.sum(self.__hessian(y2[:, 0], y2[:, 1]))
        f = lambda g, h: g**2 / (h + self.lamda)
        # print(f(Gl, Hl))
        # print(f(Gr, Hr))
        # print(f(Gl + Gr, Hl + Hr))
        # print()
        return 0.5 * ( f(Gl, Hl)+ f(Gr, Hr) - f(Gl+Gr, Hl+Hr) )

    def _leaf_value_calculation(self, y):

        return -np.sum(self.__gradient(y[:,0], y[:, 1])) / (np.sum( self.__hessian(y[:,0], y[:,1]) )+ self.lamda)

    def __gradient(self, y, y_pre):
        '''
        :param y: true label
        :param y_pre:  prediction from m-1 trees
        :return:
        '''
        return y_pre - y

    def __hessian(self, y, y_pre):
        return np.ones_like(y)  #


class XGBoost():
    def __init__(self, n_estimate=30, max_depth=2, min_sample_leaf=4):
        self.estimators = [XGBoostTree(min_sample_leaf=min_sample_leaf, max_depth=max_depth) for i in range(n_estimate)]

    def fit(self, X, y, feature_type):
        y_pre = np.zeros_like(y)
        y_y_pre = np.concatenate([y, y_pre], axis=1)
        for clf in self.estimators:
            clf.fit(X, y_y_pre, feature_type)
            y_pre += clf.predict(X)
            y_y_pre = np.concatenate([y, y_pre], axis=1)

    def predict(self, X):
        y_pred = 0
        for clf in self.estimators:
            y_pred += clf.predict(X)
        return y_pred


if __name__ == '__main__':
    data = datasets.load_boston()
    X = data.data
    y = data.target[:, None]
    clf = XGBoost(max_depth=3)
    feature_type = [0 for i in range(X.shape[1])]
    s = time.time()
    clf.fit(X, y, feature_type)
    e = time.time()
    print(e - s)
    print(np.concatenate([clf.predict(X), y], axis=1)[:30])
    print( np.mean( (y - clf.predict(X))**2) )