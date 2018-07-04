import numpy as np

import kdt


def random_kd_tree(N, D, maxleaf):
    # data matrix  
    ids = np.arange(N)
    X = np.random.uniform(0.0, 1.0, (N,D))
    kdtree = kdt.KDTBuilder(X, ids, maxleaf).build_tree()
    return kdtree


def test_kdf():

    N = 1000
    D = 20

    kdtree = random_kd_tree(N, D, maxleaf=3)

    for i in range(N):
        xi = kdtree.X[i,:]
        leaf = kdtree.find_leaf(xi)
        assert i in leaf.contents


def test_generate_knn():

    N = 1000
    D = 20

    kdtree = random_kd_tree(N, D, maxleaf=20)

    print(kdtree.generate_knn(5))

