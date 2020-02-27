# 目录

* Intuition
* 算法流程
* 代码编写
* 练习&真题
* Reference

# Intuition

Kmeans是我知道的最简单的无监督模型了。因为它无非就是这三个步骤：

* 初始化K个center

* 求出每个样本距离自己最近的center
* 通过归类到每个center的样本，求平局值来更新该center的值
* 不断重复2-3直到收敛

上面步骤中的第二步比较直观，就是越靠近那个center，这个样本就越属于这个center，理解起来非常直观。但是对第三点的理解就比较疑惑了，为什么只是简单地求平均值呢？为了解答这个疑问，我们需要从数学的角度来探讨Kmeans算法。

首先先给出Kmeans的目标函数：
$$
J(r, \mu) = \sum_n^N \sum_k^K r_{n,k} (x_n - \mu_k) ^2
$$
简单说一下上面出现的符号吧，$x_n$表示第n个样本，$\mu_k$表示第k个center，$r_{n,k}$只能取0或者是1，并且$r$每一行只能有一个数是1，其余都是0。

该目标函数有两个需要优化的目标，一个是$r$，另一个是$\mu$，要求出这两个变量的最优结果，我们使用交替优化的方法。

首先，我们固定$\mu$的值，求解$r$，对于每一个$x_n$，我们得到下面这个式子:




