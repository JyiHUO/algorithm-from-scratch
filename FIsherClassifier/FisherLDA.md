# 目录

* Intuition
* 算法步骤
* 代码编写
* Reference

# Intuition

Fisher LDA其实是一个非常容易理解的模型。一句话概括这个模型：将数据降到一维，并且让降维后的数据也能尽可能地区分开来。为了让上面这句话归纳成数学语言，我们使用两条规则来构造Fisher LDA模型。

**第一条规则**：我们需要让投影后数据的平均值尽可能隔得足够远。

假设我们有两个不同种类的数据$X_1$ 和 $X_2$。他们的平均值分别为(假设他们都有N个点):
$$
m_1 = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

$$
m_2 = \frac{1}{N}\sum_{j=1}^{N} x_j
$$

我们需要求的权重是W，那么投影后的平均值如下所示：
$$
\mu_1 = m_1^T w
$$

$$
\mu_2 = m_2^T w
$$

两个点之间的距离是：
$$
D = (\mu_1 - \mu_2)^2
$$
为了尽可能区分投影后的点，我们需要最大化投影后均值之间的距离。但是我们发现有个问题，我们只需要不断地增大W的数值就能够让D不断地增大，这样W就得不到唯一解了，所以我们需要增加限制条件，所以我们的目标函数如下所示：
$$
\max_w  D
$$

$$
s.t. \quad \sum_i^c  w_i = 1
$$

上面的约束最大化问题我们可以通过朗格朗日乘子法来求解（读者感兴趣可以自行求解）。但是上面的目标函数会存在这么一个问题，我们无法控制降维后数据的离散程度，我们希望降维后的数据越集中越好。转换成数学语言是：不同种类的数据投影后，其方差越小越好。

**第二条规则：**投影后数据的方差越小越好。

我们把投影后数据的方差表示出来：
$$
S_1 = \frac{1}{N} \sum_i^N (x_i^Tw - \mu_1)^2
$$

$$
S_2 = \frac{1}{N} \sum_j^N(x_j^Tw - \mu_2)^2
$$

我们试着结合这两条规则：
$$
J(W) = \frac{D}{S_1 + S_2}
$$
上面部分越大越好，下面部分越小越好。同时W因为分子分母同阶的关系，变成一个唯一值（读者可认真思考为什么）。我们需要最大化目标函数：
$$
\max_w J(w)
$$
因为该目标函数是一个二次函数，我们只需要对W求导，让导数为0就可以求出极值点。此时W的表示方式如下所示：
$$
(w^T S_B w)S_w w = (w^TS_w w)S_Bw
$$

$$
S_B = (m_1 - m_2)(m_1 - m_2)^T
$$

$$
S_w = \frac{1}{N} \sum_i(x_i - m_1)(x_i - m_1)^T + \frac{1}{N} \sum_j(x_j - m_2)(x_j - m_2)^T
$$

我们对上面式子中的第一个进行改写，变成：
$$
w = \alpha * S_w^{-1} (m_1 - m_2)
$$
所以w向量与右边的一项是在同一个方向上的，而且我们只需要知道w的方向就可以了，不需要准确知道该直线的bias（读者可在草稿纸上作图验证）。所以该权重求解完毕。

**以上笔记均有视频的推导过程中摘抄出来。**

# 算法步骤

* 求出不同类别数据的平均值M
* 求出$S_w$
* 根据式子:$S_w^{-1}(m_1 - m_2)$求出w

# 代码编写

有一个小地方要注意，在数据的预测部分，你需要通过一维高斯分布来估算新数据点的概率。（这里就当留个作业吧）

```python
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
```



# Reference

* Bishop pattern recognition and machine learning 2006

