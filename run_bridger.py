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

    P = 50
    N = 40000
    D = 20

    # distributed build of random data by generating "P" random seeds
    # on the master and distributing these seeds to the workers
    seeds = sc.parallelize([(i, np.random.randint(0,1000000)) for i in range(P)])
    n = N // P
    def build_points(data):
        batch, seed = data
        np.random.seed(seed)
        for j in range(n):
            yield bigkdf.DataPoint(batch*n+j, np.random.uniform(0.0, 1.0, D))

    points = seeds.flatMap(build_points)
    points = points.persist(StorageLevel.MEMORY_ONLY_SER)

    print(f"Number of items = {points.count()}")

    params = bigkdf.KDFParams(
        N=N,
        numtrees=4,
        maxnode=50,
        avgpart=5000,
        samplefactor=10,
    )

    part_trees = bigkdf.build_partition_trees(points, params)

    print("---------------------------")
    print("Build Partition Trees:")
    print(part_trees)

    mapped_points = bigkdf.map_to_subtrees(points, part_trees, params)
    subtrees = bigkdf.build_subtrees(mapped_points, params)

    print("---------------------------")
    print("Generated Subtrees:")

    subtrees.map(lambda item: print(item[0], item[1])).count()


    graph = bigkdf.build_knn_graph(subtrees, k=10)

    print("---------------------------")
    print("KNN Graph:")

    graph.map(print).count()



if __name__ == "__main__":
    main()

