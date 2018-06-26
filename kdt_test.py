import numpy as np

import kdt


def test_kdf():

    N = 1000
    D = 20

    # data matrix  
    ids = np.arange(N)
    X = np.random.uniform(0.0, 1.0, (N,D))

    kdtree = kdt.KDTBuilder(X, ids, maxleaf=3).build_tree()

    for i in range(N):
        xi = X[i,:]
        leaf = kdtree.find_leaf(xi)
        assert i in leaf.contents
