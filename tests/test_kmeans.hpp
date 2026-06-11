#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>
#include <iostream>

inline void test_kmeans_clustering() {
    std::cout << "  ⚙️  Initializing K-Means Unsupervised Cluster Verification..." << std::endl;

    // Create 4 samples with explicit multi-dimensional coordinate grouping space
    // 2 rows around coordinate 1.0f, 2 rows around coordinate 100.0f
    std::vector<float> data = {
        1.0f, 1.2f,
        1.5f, 1.1f,
        99.0f, 100.5f,
        101.2f, 98.9f
    };
    MLForge::Core::Matrix<float> X(4, 2, data);

    // Fit a K=2 clustering algorithm
    MLForge::Unsupervised::KMeans<float> cluster_engine(2, 50);
    cluster_engine.fit(X, 42);

    std::vector<size_t> labels = cluster_engine.predict(X);

    // Verify that the first two items belong to the same cluster group, 
    // and the last two items belong to the other cluster group
    assert(labels[0] == labels[1]);
    assert(labels[2] == labels[3]);
    assert(labels[0] != labels[2]);

    std::cout << "  ✅ K-Means Clustering Centroid Segmentations Passed!" << std::endl;
}