#!/usr/bin/env python3

import numpy as np

from pyspark import SparkContext, SparkConf, StorageLevel

import bigkdf


def main():
    conf = (SparkConf()
        .setMaster("local[6]")
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
        samplefactor=50,
    )

    # params = bigkdf.KDFParams(
    #     N=1000000,
    #     D=50,
    #     numtrees=2,
    #     maxnode=50,
    #     avgpart=10000,
    #     samplefactor=50,
    # )

    points = bigkdf.generate_points(sc, params, P=50)
    #points = points.persist(StorageLevel.MEMORY_ONLY_SER)

    graph = bigkdf.run_knn_process(points, params)


if __name__ == "__main__":
    main()

