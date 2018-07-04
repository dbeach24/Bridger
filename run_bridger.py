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

    N = 40000
    D = 20

    indices = sc.parallelize(range(N))
    features = indices.map(
        lambda id: bigkdf.FeatureData(id, np.random.uniform(0.0, 1.0, D))
    ).persist(StorageLevel.MEMORY_ONLY_SER)

    print(f"Number of items = {features.count()}")

    params = bigkdf.KDFParams(
        N=N,
        numtrees=4,
        maxnode=50,
        avgpart=5000,
        samplefactor=10,
    )

    part_trees = bigkdf.build_partition_trees(features, params)

    print("---------------------------")
    print("Build Partition Trees:")
    print(part_trees)

    mapped_features = bigkdf.map_to_subtrees(features, part_trees, params)
    subtrees = bigkdf.build_subtrees(mapped_features, params)

    print("---------------------------")
    print("Generated Subtrees:")

    subtrees.map(lambda item: print(item[0], item[1][1])).count()


if __name__ == "__main__":
    main()

