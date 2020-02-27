from DecisionTree.decision_tree import DecisionTree
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import time
from DecisionTree.decision_tree_parallel import DecisionTreeParallel

class DecisionTreeRegression(DecisionTree):
    def __init__(self, max_depth=float("inf"), min_sample_leaf=4):
        super().__init__(min_sample_leaf=min_sample_leaf, max_depth=max_depth)

    def _impurity_calculation(self, y, y1, y2):
        return self.__cal_variance(y) - \
               (len(y1) / len(y)) * self.__cal_variance(y1) - \
               (len(y2)/len(y)) * self.__cal_variance(y2)

    def _leaf_value_calculation(self, y):
        # return the mean of the label
        return np.mean(y.flatten())

    def __cal_variance(self, y):
        return (y - np.mean(y)).T @ (y - np.mean(y)) / float(len(y))


if __name__ == '__main__':
    data = datasets.load_boston()
    X = data.data
    y = data.target[:, None]
    feature_type = [0 for i in range(X.shape[1])]
    clf = DecisionTreeRegression(max_depth=6)  # 15.3
    s = time.time()
    clf.fit(X, y, feature_type)
    e = time.time()
    print(e - s)
    print(np.concatenate([clf.predict(X), y],axis=1 )[:30] )
    print( np.mean( (y - clf.predict(X))**2))