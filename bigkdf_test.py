import numpy as np

from pyspark import SparkContext, SparkConf, StorageLevel

import bigkdf


def make_spark_context():
    conf = (SparkConf()
        .setMaster("local[2]")
        .setAppName("Bridger")
        .set("spark.executor.memory", "1g")
        .set("spark.driver.memory", "1g")
    )
    sc = SparkContext(conf=conf)
    return sc


sc = make_spark_context()


def test_generate_points():
    params = bigkdf.KDFParams(
        N=361,
        D=7,
        numtrees=4,
        maxnode=50,
        avgpart=4000,
        samplefactor=10,
    )

    features = bigkdf.generate_points(sc, params, P=10)
    assert features.count() == params.N
    assert len(features.top(1)[0].x) == params.D


def test_build_forest():
    params = bigkdf.KDFParams(
        N=40000,
        D=20,
        numtrees=4,
        maxnode=50,
        avgpart=4000,
        samplefactor=10,
    )

    features = bigkdf.generate_points(sc, params, P=10)
    features = features.persist(StorageLevel.MEMORY_ONLY_SER)

    print(features.count())

    part_trees = bigkdf.build_partition_trees(features, params)
    print(part_trees)

    mapped_features = bigkdf.map_to_subtrees(features, part_trees, params)
    subtrees = bigkdf.build_subtrees(mapped_features, params)

    subtrees.map(print).count()

