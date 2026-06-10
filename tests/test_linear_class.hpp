#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>
#include <vector>

inline void test_logistic_regression_classification() {
    std::cout << "  ⚙️  Initializing Logistic Regression (Linear Class) Boundary Test..." << std::endl;

    // 1. Generate a small, perfectly linearly separable dataset
    // We have 4 samples with 1 feature each.
    // Low values (0.2, 0.9) belong to Class 0
    // High values (4.1, 5.5) belong to Class 1
    std::vector<float> x_features = {0.2f, 0.9f, 4.1f, 5.5f};
    MLForge::Core::Matrix<float> X(4, 1, x_features);

    std::vector<float> y_labels = {0.0f, 0.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_labels);

    // 2. Instantiate the model with a healthy learning rate and enough iterations
    MLForge::Supervised::LogisticRegression<float> model(0.2f, 1500);

    // 3. Train the model on the features and labels
    model.fit(X, y);

    // 4. Get binary decisions (should be hard thresholds: 0 or 1)
    MLForge::Core::Matrix<float> predictions = model.predict(X);

    // 5. Assertions: Check that the classification decisions match the targets perfectly
    assert(predictions(0, 0) == 0.0f);
    assert(predictions(1, 0) == 0.0f);
    assert(predictions(2, 0) == 1.0f);
    assert(predictions(3, 0) == 1.0f);

    // 6. Probability Test: Ensure probabilities match their respective regions
    MLForge::Core::Matrix<float> probas = model.predict_proba(X);
    assert(probas(0, 0) < 0.5f); // Class 0 should have a probability under 50%
    assert(probas(3, 0) > 0.5f); // Class 1 should have a probability over 50%

    std::cout << "  ✅ Logistic Regression Classification Passed! (Boundary set perfectly)" << std::endl;
}