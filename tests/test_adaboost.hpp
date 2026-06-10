#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>

inline void test_adaboost_classifier() {
    std::cout << "  ⚙️  Initializing AdaBoost Sequential Weighting Test..." << std::endl;

    std::vector<float> x_features = {
        0.1f, 1.2f,
        0.4f, 1.5f,
        3.9f, 0.1f,
        4.2f, 0.3f
    };
    MLForge::Core::Matrix<float> X(4, 2, x_features);
    std::vector<float> y_labels = {0.0f, 0.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_labels);

    // 10 boosting iterations
    MLForge::Supervised::AdaBoostClassifier<float> adaboost(10, 1.0f);
    adaboost.fit(X, y);

    MLForge::Core::Matrix<float> preds = adaboost.predict(X);

    assert(preds(0, 0) == 0.0f);
    assert(preds(3, 0) == 1.0f);

    std::cout << "  ✅ AdaBoost Classifier Passed!" << std::endl;
}