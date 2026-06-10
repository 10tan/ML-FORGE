#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>
#include <vector>

inline void test_linear_regression_fit() {
    std::cout << "  ⚙️  Initializing Linear Regression Convergence Test..." << std::endl;

    // 1. Generate clean synthetic dataset tracking y = 3x + 10
    // Inputs (X): 1.0, 2.0, 3.0, 4.0, 5.0
    std::vector<float> x_features = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    MLForge::Core::Matrix<float> X(5, 1, x_features);

    // Targets (y): 13.0, 16.0, 19.0, 22.0, 25.0
    std::vector<float> y_targets = {13.0f, 16.0f, 19.0f, 22.0f, 25.0f};
    MLForge::Core::Matrix<float> y(5, 1, y_targets);

    // 2. Instantiate the model with a safe learning rate and enough iterations to converge
    MLForge::Supervised::LinearRegression<float> model(0.03f, 1500);

    // 3. Train the model
    model.fit(X, y);

    // 4. Extract parameters
    float trained_weight = model.weights()(0, 0);
    float trained_bias = model.bias();

    // 5. Assertions: Verify results are within a tiny margin of error (0.01) of the targets
    // Target Weight = 3.0, Target Bias = 10.0
    assert(std::abs(trained_weight - 3.0f) < 0.01f);
    assert(std::abs(trained_bias - 10.0f) < 0.01f);

    // 6. Inference Test: Predict on x = 6.0 (Expected y = 3*(6) + 10 = 28.0)
    std::vector<float> unseen_x = {6.0f};
    MLForge::Core::Matrix<float> X_test(1, 1, unseen_x);
    MLForge::Core::Matrix<float> prediction = model.predict(X_test);

    assert(std::abs(prediction(0, 0) - 28.0f) < 0.01f);

    std::cout << "  ✅ Linear Regression Convergence Passed! (W~3.0, B~10.0)" << std::endl;
}