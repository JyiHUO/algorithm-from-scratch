from DecisionTree.decision_tree import DecisionTree
from DecisionTree.decision_tree_parallel import DecisionTreeParallel
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import time


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=float("inf"), min_sample_leaf=4):
        super().__init__(min_sample_leaf=min_sample_leaf, max_depth=max_depth)

    def _impurity_calculation(self, y, y1, y2):
        '''
        :param y: y in father node
        :param y1:  y in left child
        :param y2:  y in right child
        :return:
        '''
        return self.__id3(y, y1, y2)
        # return self.__gini(y, y1, y2)
        # return self.__c4_5(y, y1, y2)

    def _leaf_value_calculation(self, y):
        # return the most frequent label
        return np.argmax(np.bincount(y.flatten()))

    def __id3(self, y, y1, y2):
        return self.__cal_entropy(y) - \
               (len(y1) / len(y) * self.__cal_entropy(y1) + len(y2) / len(y) * self.__cal_entropy(y2))

    def __gini(self, y, y1, y2):
        def cal_gini(y):
            row = len(y)
            res = 0
            for u in np.unique(y):
                n = len(y[y==u])
                res += (n/row)**2
            return 1 - res
        return  -(len(y1)/len(y)) * cal_gini(y1) - (len(y2)/len(y)) * cal_gini(y2)

    def __c4_5(self, y, y1, y2):
        t = self.__cal_entropy(y)
        return (t - (len(y1) / len(y) * self.__cal_entropy(y1) + len(y2) / len(y) * self.__cal_entropy(y2)))/t


    def __cal_entropy(self, y):
        row = len(y)
        res = 0
        for u in np.unique(y):
            n = len(y[y == u])
            res += -(n / row) * np.log2(n / row)
        return res


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target[:, None]
    feature_type = [0 for i in range(X.shape[1])]
    clf = DecisionTreeClassifier(max_depth=3)
    s = time.time()
    clf.fit(X, y, feature_type)
    e = time.time()
    print(e - s)
    print(np.concatenate([clf.predict(X), y],axis=1 ) )
    print(accuracy_score(y.flatten(), clf.predict(X)))
