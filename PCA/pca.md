# 目录

* intuition
* 算法流程
* 代码编写
* 练习&考试真题
* 答案

# intuition

PCA是一个非常容易理解的无监督模型。一句话概括就是：最大化降维后数据的方差。接下来我们来看一下，怎么把我说的话转换成数学语言。

**将数据降到一维**

为了简化问题，我们从最简单的情况出发，首先计算出投影后数据的方差，假设我们的数据是X，要求的投影向量权重是w：
$$
S = \frac{1}{N}\sum_i^N (x_i^Tw - \mu_1)^2
$$

$$
\mu_1 = \frac{1}{N} \sum_i^N x_i^T w
$$

接下来我们需要最大化投影后的方差，同时稍微简化一下上面的式子：
$$
max_w S = max_w w^T S' w
$$

$$
S' = \frac{1}{N} \sum_i^N (x_i - m)(x_i - m)^T
$$

$$
m = \frac{1}{N} \sum_i^N x_i^T
$$



但是我们发现，w的值越大，S就越大，这样就求解不出来w的唯一值了，所以我们需要增加限制条件：
$$
max_w w^T S' w
$$

$$
s.t. \quad w^Tw = 1
$$

求解上面这个约束最优化问题需要使用拉格朗日乘子法来构造我们的优化目标：
$$
J(w) = w^T S' w + \alpha*(1 - w^T w)
$$
新构造的函数是二次型的函数，之间对w求导能够得到最优值：
$$
S' w = \alpha* w
$$
仔细观察上面这个式子，这是线性代数里面求特征值的结构，我们接着求解，因为:
$$
w^T w = 1
$$
所以，左乘一个WT:
$$
w^TS'w = \alpha
$$
可以看到左边这个式子就是我们想要最大化的，也就是说要想得到最优的w，我们只需要对矩阵S'进行特征值分解，然后挑选一个特征值最大的特征向量做为我们的w就求解完毕了。

**将数据降到多维**

上面举了一个特例，接下来我们让PCA变得更加general。将C维的数据降到N维，且N << C。要求解该问题我们需要使用数学归纳法。我们需要归纳的问题是：wn仍然是第n大的特征向量。

* 当N = 1时，上面我们已经证明了，数据降到一维，也就是N = 1的时候是成立的。

* 假设当N = n时，假设成立，也就是说：w 1 ... N是按顺序排列的topN个特征向量。

* 证明当N = n +1时，我们的假设成立。

首先我们需要写出我们需要优化的目标，以及一系列的约束条件：
$$
max_w w_{n+1}^T S' w_{n+1}
$$

$$
w_{n+1}^T w_{n+1} = 1
$$

$$
w_i^T w_{n+1} = 0 \quad i=1...n
$$

简单地解释一下上面的三条公式

* 第一条：我们的数据投影到第n+1这个特征向量上，我们需要最大化这些投影的方差
* 第二条：我们需要避免这个向量取无穷大
* 第三条：特征向量之间需要保持相互正交

我们使用拉格朗日乘子法来构造我们的目标函数：
$$
J(w) = w_{n+1}^T S' w_{n+1} + \alpha * (1 - w_{n+1}^T w_{n+1}) + \sum_i^n \beta_i * w_i^T w_{n+1}
$$
该函数是二次型的，求w求偏导数可以求出极值点：
$$
0 = 2*S'w_{n+1} - 2*\alpha*w_{n+1} + \sum_i^n \beta_i * w_{i}
$$
两边依次乘以 wi，求得：
$$
\beta_i = 0 \quad i=1...n
$$
所以最后我们求得：
$$
w^T_{n+1}S'w_{n+1} = \alpha
$$
我们需要最大化的目标跟左边的式子是一样的，也就是说，我们需要找一个特征值，这个特征值是最大的。并且根据约束条件来看，这个特征值跟之前的都不一样。所以只能是Top n+1 的特征值。

**总结：**

在最开始，我们将数据降维到一维后，发现我们需要求的目标是协方差（S'）的特征值。然后拓展到多维后，使用数学归纳法求出了，我们实际要求的是Top N的特征值。

# 算法流程

求出数据的协方差S'

求出S'的特征值，并取Top N个特征

# 代码编写

为了代码运行高效性，使用了svd来编写代码，基本的数学推导在下面答案部分中给出：

```python
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, n):
        '''
        n: dimention reduce
        '''
        self.w = None # (col, n)
        self.n = n

    def train(self, X):
        '''
        X: (row, col)
        '''

        new_x = (X - np.mean(X, 0)) / X.shape[0]**0.5

        U, sigma, VT = np.linalg.svd(new_x)

        self.w = VT.T[:, :self.n]

    def predict(self, X):
        return X @ self.w


if __name__ == '__main__':

    # data preparation
    m1 = np.array([-2, 3])
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
    clf = PCA(n=1)
    clf.train(X)
    new_x = clf.predict(X=X)


    # visualization
    plt.scatter(X[:, 0], X[:, 1], c="y")
    x = np.linspace(-4, 4, 100)
    print(clf.w)
    y = x * (clf.w[1] / clf.w[0])  # for W
    plt.scatter(x, y, c="g", s=5)
    plt.show()
```



# 练习&考试真题

* 为什么把数据降维之后再分类，效果会变差？试着使用画图来说明。（真题）
* 当特征数目远大于样本数目时，我们要怎么改写pca呢？
* 思考SVD与PCA的关系，为什么代码中要除以根号N（结果有可能不一致，思考为什么）。（真题）
* 自己推导一遍pca

# 答案

如何把SVD用到PCA中？

首先，先给出SVD的公式：
$$
SVD(X) = U\sum V^T
$$
我们看回去PCA最后求得的式子：
$$
S'w = \alpha w
$$
我们把权重写成矩阵的形式（读者必须在草稿纸上写一下整个过程哈）：
$$
S’ =  W \sum W^T
$$
右边的中间部分是: 对角线是特征值且其他值为0的方阵。我们结合前面推导的式子将上面的式子整理一下：
$$
\frac{(X - m)^T}{\sqrt{N}}\frac{(X - m)}{\sqrt{N}} = A^TA =   W \sum W^T
$$
对A矩阵进行SVD得：
$$
SVD(A)= U \sum' V^T
$$
将上式代入上上式得：
$$
V\sum' U^T U\sum' V^T = V \sum'^2 V^T = W\sum W^T
$$
所以我们不需要求出协方差矩阵，就可以求出W的值。

