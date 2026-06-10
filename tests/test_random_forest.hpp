#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>

inline void test_random_forest_classifier() {
    std::cout << "  ⚙️  Initializing Random Forest Consensus Test..." << std::endl;

    std::vector<float> x_features = {
        0.5f, 0.2f, // Class 0
        0.8f, 0.6f, // Class 0
        8.2f, 9.1f, // Class 1
        7.4f, 8.5f  // Class 1
    };
    MLForge::Core::Matrix<float> X(4, 2, x_features);
    std::vector<float> y_labels = {0.0f, 0.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_labels);

    // 5 Trees, Max Depth 3
    MLForge::Supervised::RandomForestClassifier<float> forest(5, 3, 2);
    forest.fit(X, y);

    MLForge::Core::Matrix<float> preds = forest.predict(X);

    assert(preds(0, 0) == 0.0f);
    assert(preds(2, 0) == 1.0f);

    std::cout << "  ✅ Random Forest Classifier Passed!" << std::endl;
}