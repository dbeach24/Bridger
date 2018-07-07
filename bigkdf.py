#!/usr/bin/env python3

from collections import namedtuple

import numpy as np

import kdt


KDFParams = namedtuple("KDFParams", [
    "N",            #< num documents in collection (may be approx)
    "numtrees",     #< num trees in forest
    "maxnode",      #< max items per leaf node
    "avgpart",      #< avg items per partition
    "samplefactor", #< number of samples per partition when building part tree
])

FeatureData = namedtuple("FeatureData", [
    "id",           #< (int) unique identifier
    "x",            #< ndarray[float, 1-D] feature vector
])

FeatureMatrix = namedtuple("FeatureMatrix", [
    "ids",          #< array of unique identifiers
    "X",            #< ndarray[float, (NxD)] feature vectors
])


def build_partition_trees(features, params):
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
    sample = features.takeSample(withReplacement=True, num=nsamples)
    vecs = [feat.x for feat in sample]

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


def map_to_subtrees(features, part_trees, params):
    """
    Given the list of partitioning trees, determine which feature
    vectors map to which subtrees.  The partitioning tree structure
    is distributed to each of the workers so that subtree assignments
    may occur in parallel.
    """

    # use a broadcast variable to distribute the partition trees
    # to all workers
    sc = features.context
    trees = sc.broadcast(part_trees)

    def subtreemapper(feature):
        """
        For each subtree in the forest to be built,
        determine the tree number and partition number within that tree
        for the given features vector.

        Note that each feature vector maps to exactly one partition
        within each tree, so there are N * numtrees results.
        """
        x = feature.x
        for treeid, parttree in enumerate(trees.value):
            subtreeid = parttree.find_leaf(x).id
            key = (treeid, subtreeid)
            yield (key, feature)

    mapped_features = features.flatMap(subtreemapper)

    return mapped_features


def build_subtrees(mapped_features, params):
    """
    With subtree assignments determined for each data point,
    group the data by subtree, building the subtree structures
    in parallel across the workers.
    """

    def create_combiner(feat):
        ids = [feat.id]
        X = [feat.x]
        return FeatureMatrix(ids, X)

    def merge_value(mat, feat):
        mat.ids.append(feat.id)
        mat.X.append(feat.x)
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

    subtrees = mapped_features.combineByKey(
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
    # a unique list of best identified neighbors for each feature
    knn = knn1.reduceByKey(combine_knn)

    return knn


