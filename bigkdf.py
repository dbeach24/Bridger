#!/usr/bin/env python3

from collections import namedtuple

import numpy as np

import kdt


KDFParams = namedtuple("KDFParams", [
    "N",            #< num documents in collection (may be approx)
    "D",            #< num dimensions in data
    "numtrees",     #< num trees in forest
    "maxnode",      #< max items per leaf node
    "avgpart",      #< avg items per partition
    "samplefactor", #< number of samples per partition when building part tree
])

DataPoint = namedtuple("DataPoint", [
    "id",           #< (int) unique identifier
    "x",            #< ndarray[float, 1-D] data vector
])

DataMatrix = namedtuple("DataMatrix", [
    "ids",          #< array of unique identifiers
    "X",            #< ndarray[float, (NxD)] (N D-dimensional points)
])


def generate_points(sc, params, P=50):
    """
    Distributed build of random data by generating "P" random seeds
    on the master and distributing these seeds to the workers
    """
    N, D = params.N, params.D
    n, s = divmod(N, P)

    seeddata = []
    pos = 0
    for p in range(P):
        end = pos + n
        if p < s: end += 1
        seed = np.random.randint(0,10000000)
        seeddata.append((pos, end, seed))
        pos = end

    def build_points(data):
        start, end, seed = data
        np.random.seed(seed)
        for i in range(start, end):
            yield DataPoint(i, np.random.uniform(0.0, 1.0, D))

    points = sc.parallelize(seeddata).flatMap(build_points)
    return points


def build_partition_trees(points, params):
    """
    Build partition trees locally by taking a sample of the
    data.  Note that this only builds the "top levels" of each
    tree.  The lower branches and leaves will be populated later.
    """

    # approximate number of partitions
    nparts = params.N / params.avgpart

    # how many binary splits are needed to create these partitions
    nsplits = int(np.ceil(np.log2(nparts)))

    # recompute number of partitions based on binary splitting
    nparts = 2**nsplits

    # how many total samples do we need to expect 10 samples per partition?
    nsamples = nparts * params.samplefactor

    # collect sample vectors locally
    # (note that we don't care about their identities, as we're just
    # trying to get a representation of the data distribution)
    sample = points.takeSample(withReplacement=True, num=nsamples)
    vecs = [point.x for point in sample]

    # build the sample data matrix
    X = np.vstack(vecs)

    def build_partition_tree():
        # build the partition tree
        return kdt.KDTBuilder(X,
            maxleaf=params.samplefactor,
            add_contents=False
        ).build_tree()

    numtrees = params.numtrees
    return [build_partition_tree() for i in range(numtrees)]


def map_to_subtrees(points, part_trees, params):
    """
    Given the list of partitioning trees, determine which points
    map to which subtrees.  The partitioning tree structure is
    distributed to each of the workers so that subtree assignments
    may occur in parallel.
    """

    # use a broadcast variable to distribute the partition trees
    # to all workers
    sc = points.context
    trees = sc.broadcast(part_trees)

    def subtreemapper(point):
        """
        For each subtree in the forest to be built, determine the
        tree number and partition number within that tree for the
        given point.

        Note that each point maps to exactly one partition
        within each tree, so there are N * numtrees results.
        """
        x = point.x
        for treeid, parttree in enumerate(trees.value):
            subtreeid = parttree.find_leaf(x).id
            key = (treeid, subtreeid)
            yield (key, point)

    mapped_points = points.flatMap(subtreemapper)

    return mapped_points


def build_subtrees(mapped_points, params):
    """
    With subtree assignments determined for each data point,
    group the data by subtree, building the subtree structures
    in parallel across the workers.
    """

    def create_combiner(point):
        ids = [point.id]
        X = [point.x]
        return DataMatrix(ids, X)

    def merge_value(mat, point):
        mat.ids.append(point.id)
        mat.X.append(point.x)
        return mat

    def merge_combiners(mat1, mat2):
        mat1.ids.extend(mat2.ids)
        mat1.X.extend(mat2.X)
        return mat1

    def build_subtree(mat):
        subtree = kdt.KDTBuilder(
            np.asarray(mat.X),
            np.asarray(mat.ids),
            maxleaf=params.maxnode,
        ).build_tree()
        return subtree

    subtrees = mapped_points.combineByKey(
        create_combiner,
        merge_value,
        merge_combiners
    ).mapValues(build_subtree)

    return subtrees


def build_knn_graph(subtrees, k):
    """
    Use the built subtrees to create an initial KNN graph.
    Points which reside in the same leaf node of a subtree
    are evaluated pairwise as neighbor candidates.  These
    neighbors are evaluated locally first, keeping the k
    best neighbors.  Then, neighbor lists are collected and
    reduced across the various trees to keep the k-best
    neighbors from each worker.

    The entire KNN graph is returned as an RDD.
    """

    def get_knn(item):
        """Find KNN lists by comparing points in the same leaf."""
        key, subtree = item
        ids = subtree.ids
        X = subtree.X
        for i, neighbors in subtree.generate_knn(k):
            nmap = [(ids[x], d) for x, d in neighbors]
            yield (ids[i], nmap)

    # this KNN graph contains nearest neighbors found in each tree,
    # but has redundant entries from the different trees in the forest
    knn1 = subtrees.flatMap(get_knn)

    def combine_knn(neighbors1, neighbors2):
        """Combine different KNN lists for the same point."""
        all_neighbors = neighbors1 + neighbors2
        uniq_ids = set(i for (i, d) in all_neighbors)
        all_neighbors = [(i,d) for (i,d) in all_neighbors if i in uniq_ids]
        all_neighbors.sort(key=lambda x: x[1])
        del all_neighbors[k:]
        return all_neighbors

    # combine the neighborhood lists across different trees to produce
    # a unique list of best identified neighbors for each point
    knn = knn1.reduceByKey(combine_knn)

    return knn


