import numpy as np

from pyspark import SparkContext, SparkConf, StorageLevel

import bigkdf


def test_build_forest():
    conf = (SparkConf()
        .setMaster("local[2]")
        .setAppName("Bridger")
        .set("spark.executor.memory", "1g")
        .set("spark.driver.memory", "1g")
    )
    sc = SparkContext(conf=conf)

    N = 40000
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
        avgpart=4000,
        samplefactor=10,
    )

    part_trees = bigkdf.build_partition_trees(features, params)
    print(part_trees)

    mapped_features = bigkdf.map_to_subtrees(features, part_trees, params)
    subtrees = bigkdf.build_subtrees(mapped_features, params)

    subtrees.map(print).count()

