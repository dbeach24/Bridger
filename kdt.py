from collections import namedtuple, defaultdict
import itertools
import heapq

import numpy as np


KDNode = namedtuple("KDNode", [
    "id",           #< locally unique identifier for node (int)
    "sdim",         #< dimension on which to split the data
    "sval",         #< value by which to split the data
    "left",         #< left child (KDNode or None)
    "right",        #< right child (KDNode or None)
    "contents"      #< contents (list of FeatData, for leaf nodes only!)
])


def is_leaf(node):
    """True if this KDNode is a leaf."""
    return not (node.left or node.right)


def get_counts(node):
    """Return count of (leaves, items) under the given node."""
    if is_leaf(node):
        return (1, len(node.contents) if node.contents else 0)
    else:
        lc, li = get_counts(node.left)
        rc, ri = get_counts(node.right)
        return (lc + rc, li + ri)


def _random_split(X, k=5):
    """
    Selects a split dimension and value from one of the k highest
    variance columns in X.

    Randomly selecting the split column is important to ensure that multiple
    trees are not highly correlated.  However, biasing the algorithm towards
    high-variance columns is important for performance.  According to the
    literature, k=5 is a good choice in practice.

    - X: data matrix (NxD)
    - k: number of high-variance columns from which to choose
    - returns: (split dimension, split value)
    """
    v = X.var(axis=0)
    varcols = np.argsort(v)[:k]
    sdim = varcols[np.random.randint(k)]
    sval = np.median(X[:,sdim])
    return sdim, sval


class KDTBuilder:
    """
    Build a KDTree, referencing points as the rows of matrix X.
    
    This algorithm recursively subdivides the elements of X by choosing
    a split dimension and value, and constructs a tree of values whose
    leaves have no more than maxleaf items each.

    Note that the KDTree does not store values of X directly, but
    references the rows of X.
    """

    def __init__(self, X, ids=None, maxleaf=10, add_contents=True):
        """
        - X: data matrix (NxD)
        - ids: vector of IDs (N) [default: np.arange(N)]
        - maxleaf: maximum number of points per leaf
        - add_contents: add X indices to leaves while building
        """
        N = X.shape[0]
        if ids is None:
            ids = np.arange(N)
        else:
            ids = np.asarray(ids)
        self._X = X
        self._ids = ids
        self._maxleaf = maxleaf
        self._add_contents = add_contents
        self._makeid = itertools.count(0).__next__

    def build_tree(self):
        """
        - returns: KDNode, the root of a KD tree
        """
        N = self._X.shape[0]
        indices = np.arange(N)
        root = self._build_node(indices)
        if not self._add_contents:
            self._X = None
            self._ids = None
        return KDTree(root, self._X, self._ids)

    def _build_node(self, indices):
        """
        Build nodes of a KDTree, recursively.
        """

        nodeid = self._makeid()

        N = len(indices)
        if N <= self._maxleaf:
            if self._add_contents:
                contents = list(indices)
            else:
                contents = None
            return KDNode(nodeid, None, None, None, None, contents)

        if N > 100:
            samp = np.random.choice(indices, 100, replace=True)
        else:
            samp = indices
        
        Xsamp = self._X[samp,:]

        sdim, sval = _random_split(Xsamp)

        s = self._X[indices,sdim] < sval
        idxleft = indices[s]
        idxright = indices[~s]

        # clean up storage before recursion
        Xpart = None

        left = self._build_node(idxleft)
        right = self._build_node(idxright)

        return KDNode(nodeid, sdim, sval, left, right, None)


class KDTree:

    def __init__(self, root, X, ids):
        assert isinstance(root, KDNode)
        self._root = root
        self._X = X
        self._ids = ids
        self._numleaves, self._numitems = get_counts(root)

    def __repr__(self):
        return "<KDTree with %i leaves and %i items>" % (
            self._numleaves, self._numitems)

    @property
    def root(self):
        return self._root

    @property
    def X(self):
        return self._X
    
    @property
    def ids(self):
        return self._ids

    def find_leaf(self, xi):
        """
        Find the KDNode under tree which should contain vector xi.
        """
        node = self._root
        while not is_leaf(node):
            goleft = xi[node.sdim] < node.sval
            node = node.left if goleft else node.right
        return node

    def generate_knn(self, k):
        """
        Returns KNNState results.
        Note that the IDs used are indexes into the ids array.
        """

        X = self._X

        def dist2(i, j):
            d = X[i,:] - X[j,:]
            return np.dot(d, d)

        def leafpairwiseknn(node):
            if node.left:
                yield from leafpairwiseknn(node.left)
            if node.right:
                yield from leafpairwiseknn(node.right)

            if not is_leaf(node):
                return

            items = node.contents
            knn = KNNState(k)
            for i, j in itertools.combinations(items, 2):
                d2 = dist2(i, j)
                if d2 < 1e-10:
                    print(i, j, X[i,:], X[j,:], d2)
                    assert False
                knn.add_neighbor(i, j, d2)

            yield from knn.graph.items()

        yield from leafpairwiseknn(self.root)


class KNNState:

    def __init__(self, k):
        self._k = k
        self._graph = defaultdict(list)

    @property
    def k(self):
        return self._k

    @property
    def graph(self):
        return self._graph

    def add_neighbor(self, i, j, dist):

        k = self._k

        def knnadd(neighbors, idx):
            if len(neighbors) >= k and dist > neighbors[k-1][1]:
                return
            neighbors.append((idx, dist))
            neighbors.sort(key=lambda x: x[1])
            del neighbors[k:]

        graph = self._graph
        knnadd(graph[i], j)
        knnadd(graph[j], i)

