#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>

inline void test_decision_tree_classifier() {
    std::cout << "  ⚙️  Initializing Decision Tree Boundary Test..." << std::endl;

    // Simple 2D classification problem: 
    // If Feature 0 <= 2.0 -> Class 0, else Class 1 (Feature 1 is noise)
    std::vector<float> x_features = {
        1.0f, 10.0f,  // Class 0
        1.5f, 20.0f,  // Class 0
        4.0f, 11.0f,  // Class 1
        5.5f, 19.0f   // Class 1
    };
    MLForge::Core::Matrix<float> X(4, 2, x_features);
    std::vector<float> y_labels = {0.0f, 0.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_labels);

    // Instantiate with Max Depth = 3, Min Samples Split = 2
    MLForge::Supervised::DecisionTreeClassifier<float> model(3, 2);
    model.fit(X, y);

    // Predict on the training data to confirm perfect cell partitioning
    MLForge::Core::Matrix<float> preds = model.predict(X);

    assert(preds(0, 0) == 0.0f);
    assert(preds(1, 0) == 0.0f);
    assert(preds(2, 0) == 1.0f);
    assert(preds(3, 0) == 1.0f);

    std::cout << "  ✅ Decision Tree Classifier Passed! (Gini partitioning functional)" << std::endl;
}