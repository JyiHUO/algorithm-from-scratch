# 目录

* Intuition
* 算法流程
* 代码编写
* reference
* 练习

# Intuition

**讲系前面**

今天讲一下RBF网络吧，其核心思想是RBF。一句话概括该模型：找到“支持向量“，使用”支持向量“来估算输入的数据。

个人感觉RBF net其实跟Kmean非常相似。但是其中存在的不同是rbf会利用kmean求出的每一个center，给数据估算出”距离”，然后通过加权的方法来推断结果。总的来说公式如下所示(以下是回归模型，分类模型只需要简单改一下目标函数~)：
$$
RBFnet(x) = \sum_i^m w_i exp(-\frac{ (x - c_i)^2 }{ r^2 })
$$
简单地解释一下参数：w是模型所要求解的权重。x是输入数据，c是模型的“支持向量”，r是我们需要提前设置的参数，比如说0.1。接下来我将从这几部分讲解RBF net：

* RBF的概念
* 如何寻找我们的支持向量
* 如何估算数据

**RBF的概念**

首先给出RBF的公式：
$$
RBF(x,c) = \exp(-\frac{(x - c)^2}{r^2})
$$
对上面式子最直观的理解是，对点x与点c的距离做一个平滑的变化（重点是距离公式）。

如果我们对该公式进行可视化的解释，如下图所示[1]：

![image-20191220145554756](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191220145554756.png)

第二幅图表示每个样本点都服从：$f(x) = exp(-\frac{(x - c)^2}{r^2})$分布（这个分布的累积和不为1哦），其中c是这个分布的中心，也就是图中的蓝色点。往两边走，值不断变少。那我们怎么把蓝色的点都连起来呢？比如说我们怎么求出来偏蓝色点右边一点点新的一个点（a点）呢？此时我们需要借助b这个分布和c这个分布，通过一定的权重叠加来求出来，这就是我们后面要讲到的linear 部分。

总的来说，rbf是计算中心点（c）和其他点（x）径向距离的公式。

**如何寻找我们的支持向量**

寻找我们的支持向量，我们有这么几种方法：

* Kmean：对不同标签的数据聚类找到center
* 将全部的训练样本作为“支持向量”
* 随机从训练样本抽取“支持向量”

通常在这里，支持向量的数量会比原始数据的维度要大，所以会产生数据“升维”的错觉：
$$
G_{ij} = \exp(-\frac{(X_{i} - C_j )^2}{r^2}) \quad i=1...N \quad j = 1...m
$$
其中N是N个样本，m是center的数目，X是二维的多个样本拼接的矩阵，C是对个center拼接的矩阵，G是新产生的矩阵。

**如何估算数据**

我们挑选的“支持向量”是有好坏之分的，所以我们需要给不同的"支持向量"给予不同的权重，所以我们使用linear模型来构造我们的目标函数（如果是分类问题，你可以使用交叉熵来构造你的目标函数）得到下面这个式子：
$$
L(w) = \frac{1}{2} (Gw - y)^2 + \frac{ \lambda}{2} w^Tw
$$
该函数是二次型函数，我们队w求导数得：
$$
w = (G^T G + \lambda I)^{-1}G^T y
$$

# 算法流程

* 使用kmean求出矩阵G
* 求解权重w

# 代码编写

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap


class RBFNet():
    def __init__(self, M, r=0.1, reg=0.001):
        '''
        m: 升维度后的维度
        r: rbf 的参数
        reg: 正则化参数
        '''
        self.M = M
        self.r = r
        self.reg = reg

        # setting for logistic regression
        self.lr = 0.01
        self.epoch = 100
        self.w = None
        self.b = None

    def rbf(self, x, u):
        return np.exp( - ((x - u) @ (x-u)) / self.r)

    def __cal_center(self, X):
        clf = KMeans(n_clusters=self.M)
        clf.fit(X)
        return clf.cluster_centers_

    def __sigmoid(self, z):
        return 1 / (1 + np.e ** (-z))

    def __linear_regression_train(self, G):
        self.w = np.linalg.inv(G.T @ G + self.reg*np.diag(G.shape[1])) @ G.T @ y

    def __linear_regression_predict(self, G):
        return G @ self.w

    def __logistic_regression_train(self, X, y):
        # initialize W and b
        N, D = X.shape
        y = y.reshape((N, 1))
        self.w = np.random.standard_normal(size=(D, 1))
        self.b = np.random.standard_normal(size=(1, 1))

        for i in range(self.epoch):
            # forward
            z = X.dot(self.w)
            a = self.__sigmoid(z)
            # loss = -1.0 / N * (y.T.dot(np.log(a)) + (1 - y).T.dot(np.log(1 - a)))

            # backward
            dw = X.T.dot(a - y) / N
            db = np.mean(a - y)

            # update "
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def __logistic_regression_predict(self, X, thresold = 0.5):
        z = X.dot(self.w)
        a = self.__sigmoid(z)
        # return (a > thresold).astype(int)
        return a

    def __cal_G(self, X):
        N, p = X.shape
        G = np.zeros((N, self.M))

        for i in range(N):
            for j in range(self.M):
                G[i, j] = self.rbf(X[i, :], self.C[j, :])
        return G


    def train(self, X, y):

        self.C = self.__cal_center(X)

        G = self.__cal_G(X)

        self.__logistic_regression_train(G, y)

    def predict(self, X):
        G = self.__cal_G(X)
        return self.__logistic_regression_predict(G)


if __name__ == '__main__':
    X, y = make_moons(noise=0.3, random_state=0)
    clf = RBFNet(M=10)
    clf.train(X, y)

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
```



# Reference

* https://shomy.top/2017/02/26/rbf-network/
* PMRL

# 练习

* 请使用二维和三维向量来解说rbf模型
* 从头编写代码
* 在使用rbf前，为什么需要归一化数据的每一列？