import numpy as np
from collections import deque

# two player
data = [[(0,0), (0,0),(0,0),(0,0)],
        [(0,0),(0,0),(-1,1),(-1,1)],
        [(0,1),(1,-1),(0,0),(-1,1)],
        [(0,0),(1,-1),(1,-1),(0,0)]]

# player1_matrix = np.array([[0,0,0,0],
#                            [0,0,-1,-1],
#                            [0,1,0,-1],
#                            [0,1,1,0]])
# player2_matrix = np.array([[0,0,0,0],
#                            [0,0,1,1],
#                            [1,-1,0,1],
#                            [0,-1,-1,0]])


def helper(p, data, res, indexs, player):
    if p == 1:
        v = data[tuple(indexs)]
        max_val = -10**8
        max_indexs = []
        for i in range(len(v)):
            if v[i][player] >= max_val:
                max_indexs.append(i)
        for index in max_indexs:
            indexs.append(index)
            res[tuple(indexs)] += 1
            indexs.pop()
    else:
        p-=1
        ns = data.shape[0]
        for i in range(ns):
            new_indexs = indexs[:]
            new_indexs.append(i)
            helper(p, data, res, new_indexs)

data = np.array(data)  # (ns, ns)

res = np.zeros_like(data)

num_player = len(data.shape) - 1
num_strategy = data.shape[0]

# check bug
axes = deque(list(range(0, num_player)))
for p in range(num_player):
    axes.append(axes.popleft())
    data.transpose(axes)
    helper(num_player, data, res, [], p)


