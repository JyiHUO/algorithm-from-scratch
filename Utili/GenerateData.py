import numpy as np
np.random.seed(10)


def test_clf(correct_clf, test_clf):
    '''
    :param correct_clf:
    :param test_clf: must include train method and test method
    :return:
    '''
    m1 = np.array([0, 3])
    m2 = np.array([3, 2.5])
    C1 = np.array([[1, 0], [0, 1]], np.float32)
    C2 = np.array([[1, 0], [0, 1]], np.float32)

    N = 200
    data1 = np.random.randn(N, 2)
    data2 = np.random.randn(N, 2)
    A1 = np.linalg.cholesky(C1)
    A2 = np.linalg.cholesky(C2)
    new_data1 = data1 @ A1.T + m1
    new_data2 = data2 @ A2.T + m2

    X = np.concatenate([new_data1, new_data2], axis=0)
    y = np.concatenate([np.ones(200), np.zeros(200)])

    X_test = np.random.rand(N, 2) * 6

    # train model
    clf1 = correct_clf()
    clf1.train(X, y)
    clf2 = test_clf()
    clf2.train(X, y)
    if np.sum(clf1.predict(X_test) == clf2.predict(X_test)) > 2:
        print("sample one error")

    # sample two
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
    y = np.concatenate([np.ones(200), np.zeros(200)])

    X_test = np.random.rand(N, 2) * 6

    # train model
    clf1 = correct_clf()
    clf1.train(X, y)
    clf2 = test_clf()
    clf2.train(X, y)
    if np.sum(clf1.predict(X_test) == clf2.predict(X_test)) > 2:
        print("sample two error")

    # sample three
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
    y = np.concatenate([np.ones(200), np.zeros(200)])

    X_test = np.random.rand(N, 2) * 6

    # train model
    clf1 = correct_clf()
    clf1.train(X, y)
    clf2 = test_clf()
    clf2.train(X, y)
    if np.sum(clf1.predict(X_test) == clf2.predict(X_test)) > 2:
        print("sample three error")