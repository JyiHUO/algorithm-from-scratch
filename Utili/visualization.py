import numpy as np
import matplotlib.pyplot as plt

def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)
    dc = np.linalg.det(Ci)
    num = np.exp(-0.5 * (x - m).T @ (Ci @ (x - m)))
    den = 2 * np.pi * dc
    return num / den  # this is the probability


def twoD(nx, ny, m, C):
    x = np.linspace(-7, 7, nx)
    y = np.linspace(-7, 7, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gauss2D(xvec, m, C)
    return X, Y, Z


def posteriorPlot(nx, ny, m1, C1, m2, C2, P1, P2):
    x = np.linspace(-7, 7, nx)
    y = np.linspace(-7, 7, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i, j], Y[i, j]])
            num = P1 * gauss2D(xvec, m1, C1)
            den = P1 * gauss2D(xvec, m1, C1) + P2 * gauss2D(xvec, m2, C2)
            Z[i, j] = num / den
    return X, Y, Z

def plotGaussianD(m, C, N=200, c='r'):
    N = 200
    data = np.random.randn(N, 2)
    A = np.linalg.cholesky(C)
    new_data = data @ A.T + m
    plt.scatter(new_data[:, 0], new_data[:, 1], c=c)
