import numpy as np
from skimage import io


class NMF():
    def __init__(self, rank, alpha=0.01, epoch = 20):
        self.alpha = alpha
        self.epoch = epoch
        self.rank = rank
        self.W = None
        self.H = None

    def train(self, X):
        # init
        row, col = X.shape
        v_max, v_min = np.max(X), np.min(X)
        W = np.random.rand(row, self.rank) * (v_max - v_min) + v_min
        H = np.random.rand(self.rank, col) * (v_max - v_min) + v_min

        # update
        for e in range(self.epoch):
            W = W * ( (X @ H.T)/(W @ H @ H.T) )
            H = H * ( (W.T @ X) / (W.T @ W @ H) )
        self.W = W
        self.H = H


if __name__ == '__main__':
    img_src = "https://test87983.oss-cn-shanghai.aliyuncs.com/299536/poster.jpg"
    image = io.imread(img_src)

    clf = NMF(rank=300, epoch=100)
    clf.train(image[:, :, 0])
    new_image = clf.W @ clf.H
    print(new_image)
    print(image[:, :, 0])
    new_image = new_image.astype("uint8")
    io.imshow(new_image)
    io.show()

