#!/usr/bin/env python3

import numpy as np

from pyspark import SparkContext, SparkConf, StorageLevel

import bigkdf


def main():
    conf = (SparkConf()
        .setMaster("local[4]")
        .setAppName("Bridger")
        .set("spark.executor.memory", "2g")
        .set("spark.driver.memory", "4g")
    )
    sc = SparkContext(conf=conf)

    N = 4000000
    D = 20

    indices = sc.parallelize(range(N))
    features = indices.map(
        lambda id: bigkdf.FeatureData(id, np.random.uniform(0.0, 1.0, D))
    ).persist(StorageLevel.MEMORY_ONLY_SER)

    print(features.count())

    params = bigkdf.KDFParams(
        N=N,
        numtrees=4,
        maxnode=50,
        avgpart=20000,
        samplefactor=10,
    )

    part_trees = bigkdf.build_partition_trees(features, params)

    print(part_trees)

    mapped_features = bigkdf.map_to_subtrees(features, part_trees, params)
    #mapped_features = mapped_features.cache()

    #print(mapped_features.takeSample(False, 20))

    subtrees = bigkdf.build_subtrees(mapped_features, params)

    #subtrees = subtrees.persist(StorageLevel.MEMORY_AND_DISK_SER)
    subtrees.map(print).count()


if __name__ == "__main__":
    main()

