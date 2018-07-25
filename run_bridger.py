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

    params = bigkdf.KDFParams(
        N=70000,
        D=784,
        numtrees=4,
        maxnode=50,
        avgpart=3500,
        samplefactor=10,
    )

    points = bigkdf.generate_points(sc, params, P=50)
    points = points.persist(StorageLevel.MEMORY_ONLY_SER)

    print(f"Number of items = {points.count()}")

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

    #graph.map(print).count()
    print(graph.count())



if __name__ == "__main__":
    main()

