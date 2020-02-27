import numpy as np
from collections import Counter


class Node():
    def __init__(self, leaf_value, best_feature, best_threshold, left_branch, right_branch):
        self.leaf_value = leaf_value
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.left_branch = left_branch
        self.right_branch = right_branch


class DecisionTree():
    '''
    This is the super class for Classifier tree, Regression tree and XGBRegression
    '''
    def __init__(self, min_sample_leaf =4, max_depth=float("inf"), min_impurity=1e-5):
        self.root = None
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity = min_impurity
        self.features_type = None  # [0, 1, 0, 1 ...] 1 means cate feature, 0 means continuous feature

    def fit(self, X, y, features_type):
        '''
        X: (row, col)
        y: (row, 1)
        features_type: [0, 1, 0, 1]
        '''
        assert len(y.shape) == 2
        self.features_type = features_type
        self.root = self._build(X, y)

    def predict(self, X):
        '''
        :param X: numpy array (row, col)
        :return:
        '''
        y = []
        for i in range(X.shape[0]):
            y.append(self._predict_helper(X[i], self.root))
        return np.array(y)[:, None]

    def _predict_helper(self, x, node):
        if node.leaf_value is None:
            if self.features_type[node.best_feature] == 1:
                # for cate feature
                if x[node.best_feature] == node.best_threshold:
                    return self._predict_helper(x, node.left_branch)
                else:
                    return self._predict_helper(x, node.right_branch)
            else:
                # for continuous feature
                if x[node.best_feature] < node.best_threshold:
                    return self._predict_helper(x, node.left_branch)
                else:
                    return self._predict_helper(x, node.right_branch)
        else:
            return node.leaf_value

    def _build(self, X, y, depth=0):
        '''
        X: (row, col)
        y: (row, 1)
        '''
        node = Node(None, None, None, None, None)
        row, col = X.shape

        if self.max_depth >= depth and self.min_sample_leaf < row:
            # find the best feature to split
            best_f_index = None
            best_f_gain = -10**8
            best_threshold = None
            for f_index in range(col):
                thresholds = self._generate_thresholds(X[:, f_index], self.features_type[f_index])
                for threshold in thresholds:
                    X1, y1, X2, y2 = self._divide_dataset(X, y, f_index, threshold, self.features_type[f_index])
                    gain = self._impurity_calculation(y, y1, y2)
                    if gain >= best_f_gain:
                        best_f_gain = gain
                        best_threshold = threshold
                        best_f_index = f_index
            if best_f_gain > self.min_impurity:
                X1, y1, X2, y2 = self._divide_dataset(X, y, best_f_index, best_threshold, self.features_type[best_f_index])
                node.best_threshold = best_threshold
                node.best_feature = best_f_index
                node.leaf_value = None
                del X
                node.left_branch = self._build(X1, y1, depth+1)
                node.right_branch = self._build(X2, y2, depth+1)
            else:
                # do not split anymore
                del X
                node.leaf_value = self._leaf_value_calculation(y)
            return node
        else:
            del X
            node.leaf_value = self._leaf_value_calculation(y)
            return node

    def _leaf_value_calculation(self, y):
        pass

    def _impurity_calculation(self, y, y1, y2):
        pass

    def _divide_dataset(self, X, y, feature_index, threshold, feature_type):
        if feature_type == 1:
            t = X[:, feature_index] == threshold
            return X[t],y[t], X[~t],y[~t]
        else:
            t = X[:, feature_index] < threshold
            return X[t],y[t], X[~t],y[~t]

    def _generate_thresholds(self, feature, feature_type):
        if feature_type == 1:
            return np.unique(feature)
        else:
            s = sorted(feature)
            return [ (s[i]+s[i-1])/2 for i in range(1, len(s)) ]
