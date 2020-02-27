# 目录

* 前提假设
* 算法讲解
* 算法步骤
* 代码

# 前提假设

* 该问题是二分类问题

* 每个类别的分布符合高斯分布

# 算法讲解

首先在开篇之前，把贝叶斯公式放上来，这是一切的基础啊，当然后面也会用到~：
$$
P(w|x) = \frac{P(x|w)P(w)}{P(x)}
$$

接下来我们就来看看怎么使用这条公式建模吧。首先我们简单地介绍一下什么是高斯分布的概率密度函数吧，它的数学公式是这样的：
$$
f_{X}(x_1,...,x_k) = \frac{exp(-\frac{1}{2}(x-\mu)^T C^{-1}(x-\mu)) }{\sqrt{(2\pi)^k|C|} }
$$
公式看起来可能有点抽象，我用几个例子来带你直观理解高斯分布的概率密度函数。其实该函数最重要的参数是$C$(协方差)和$\mu$(分布的中心)，当

```python
m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = np.array([[2,1], [1,2]], np.float32)
C2 = np.array([ [2,1], [1,2] ], np.float32)
```

![image-20191027105113739](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191027105113739.png)

插一句话：在这里简单介绍一下协方差吧，左对角线是变量自己的相关系数，这个数越大圈里面的点扩散得越厉害。矩阵其他的数值就是变量本身跟其他变量的相关系数，这个数越大这个方向的椭圆就越尖。当

```python
m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = np.array([[2,0], [0,2]], np.float32)
C2 = np.array([ [1.5,0], [0,1.5] ], np.float32)
```

![image-20191027105846114](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191027105846114.png)

以上是对高斯分布的一个直观认识，你可以看到上面的数据可以分成两个类别，红色代表+，绿色代表-，他们是两个不一样的高斯分布，如果来了一个蓝点，我需要判断它是属于红色这个分布还是绿色这个分布？

![image-20191027110438427](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191027110438427.png)

首先你或许会给出这样一个解释，直接算这个蓝色点到这两个分布的中心的距离，然后判断离那个中心近就属于那个不就可以了吗？但是你忽略了一个重要的条件，两个分布的的各自的协方差不一样，也就是说这个蓝点到两个分布的欧式距离不是真实的（这个后面会说到），你需要把协方差考虑进去。这时候贝叶斯出现了，我们将这个直观的问题建模成数学语言：
$$
P(w1|x) = \frac{P(w1)P(x|w1)}{P(w1)P(x|w1) + P(w2)P(x|w2)}
$$

$$
P(w2|x) = \frac{P(w2)P(x|w2)}{P(w1)P(x|w1) + P(w2)P(x|w2)}
$$

$$
P(w1|x) \le \ge P(w2|x)
$$

简单地解释一下其中的符号吧：

* $P(w1)$: 权重出现的概率，通常为先验知识
* $P(x|w1)$:输入权重后，得到样本的概率，也就是高斯分布概率密度函数
* $P(w1|x)$:输入样本，求这个属于这个权重的概率，通俗来说是样本属于这个类别的概率
* $P(w1|x) \le \ge P(w2|x)$: 输出那个类别的概率更大

我们接着中第三个式子来看：
$$
P(w1|x) \le \ge P(w2|x)
$$

$$
\frac{P(w1)P(x|w1)}{P(w1)P(x|w1) + P(w2)P(x|w2)} \le \ge \frac{P(w2)P(x|w2)}{P(w1)P(x|w1) + P(w2)P(x|w2)}
$$

$$
P(w1)P(x|w1)\le\ge P(w2)P(x|w2)
$$

假设$P(w1) = P(w2)$:
$$
\frac{exp(-\frac{1}{2}(x-\mu_1)^TC_1^{-1}(x-\mu_1)) }{\sqrt{(2\pi)^k|C_1|} } \le \ge \frac{exp(-\frac{1}{2}(x-\mu_2)^TC_2^{-1}(x-\mu_2)) }{\sqrt{(2\pi)^k|C_2|} }
$$
假设$C_1 = C_2 = \sigma^2I$:
$$
(x - \mu_1)^T (x - \mu_1) \le \ge (x - \mu_2)^T (x - \mu_2)
$$
推导到这里，你是否有种豁然开朗的感觉，比较样本属于那个类别的概率更大，其实是找到这个点到两个分布中心距离那个更短。不过这是在层层假设推导出来的，现在我们还原回去：

假设$C_1 \ne C_2$和$P(w1) \ne P(w2)$: 
$$
\alpha* (x-\mu_1)^TC_1^{-1}(x-\mu_1) \le \ge (x-\mu_2)^TC_2^{-1}(x-\mu_2)
$$
我们再欧氏距离中加了协方差矩阵，这个叫做[Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)。所以高斯分类器其实也是根据距离判断的分类器啊。看了这么多数学推导你可能会觉得眼花缭乱，反应不过来，还是那句话，结合例子来看吧：

当$C_1 = C_2 = \sigma^2 I$ 和 $P(w1) \ne P(w2)$，这时候的欧氏距离跟马氏距离是相等的，但是线往右边平移了（想想为什么？）

![image-20191027121654849](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191027121654849.png)

我们再看一个例子$C_1 \ne C_2$ 和 $P(w1) = P(w2)$，这时候画出来的是一条曲线，为什么呢？为什么上面画出来的是一条直线，下面画出来的是一条曲线呢？

![image-20191027122030021](/Users/huojunyi/Library/Application Support/typora-user-images/image-20191027122030021.png)

聪明地你可能会发现直线的直和曲跟协方差C有关，所以接下来我们需要把高斯分类器推导出来，也就是求出图中的分类边界。
$$
P(w1|x) =\frac{P(w1)P(x|w1)}{P(w1)P(x|w1) + P(w2)P(x|w2)} = \frac{1}{1 + \frac{P(w2)P(x|w2)}{P(w1)P(x|w1)}}
$$
当$P(w1) \ne  P(w2)$和$C_1 = C_2$:
$$
P(w1|x) = \frac{1}{ 1 + \frac{P(w2)}{P(w1)}  e^{ (x^T\C^{-1}x - 2\mu_2^T\C^{-1}x + \mu_2\C^{-1}\mu_2) - (x^T\C^{-1}x - 2\mu_1^T\C^{-1}x + \mu_1\C^{-1}\mu_1) } } = \frac{1}{ 1 + \alpha e^{x^Tw + b}}
$$
这个结果是不是很像sigmoid，指数部分是直线方程，所以它的分类边界肯定是直线啊。

当$P(w1) \ne  P(w2)$和$C_1 \ne C_2$:
$$
P(w1|x)  = \frac{1}{1 + \alpha e^{ x^T A x + Bx + C }}
$$
读者可以自行吧A，B，C求出来，你可以清除地看到指数部分是二次方程，所以分类边界是一条曲线。

以上就是高斯分类器的公式推导过程和一些对该分类器的直观理解，接下来我们将这些知识归纳成一步步的算法步骤。

# 算法步骤

* 分别求出训练集中正样本和负样本的协方差矩阵$C$和平均值$\mu$
* 设置参数$\alpha$, 其中$\alpha = \frac{P(w2)}{P(w1)}$
* 输出概率$P(w1|x) = \frac{1}{1 + \alpha \frac{P(x|w2)}{P(x|w1)} }$，其中$P(w1|x) = \frac{exp(-\frac{1}{2}(x-\mu_1)^TC_1^{-1}(x-\mu_1)) }{\sqrt{(2\pi)^k|C_1|} }$

# 代码

```python
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
        return np.exp(-0.2*  (((X - mu) @ np.linalg.inv(C)) * (X - mu)).sum(axis=1) ) / np.sqrt( (2 * np.pi)**k * np.linalg.det(C) )

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
```


$$

$$


