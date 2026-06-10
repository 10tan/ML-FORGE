#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>

inline void test_metrics_evaluation() {
    std::cout << "  ⚙️  Initializing Loss Metrics Evaluation Test..." << std::endl;

    // Target values
    std::vector<float> true_vals = {1.0f, 2.0f, 3.0f};
    MLForge::Core::Matrix<float> y_true(3, 1, true_vals);

    // Predicted values (off by exactly 0.5 each: diffs are -0.5, -0.5, -0.5)
    // squared diffs: 0.25, 0.25, 0.25 -> sum = 0.75 -> mean (divide by 3) = 0.25
    std::vector<float> pred_vals = {1.5f, 2.5f, 3.5f};
    MLForge::Core::Matrix<float> y_pred(3, 1, pred_vals);

    float mse = MLForge::Utils::Metrics<float>::mean_squared_error(y_true, y_pred);
    float rmse = MLForge::Utils::Metrics<float>::root_mean_squared_error(y_true, y_pred);

    // Assert within fine float limits
    assert(std::abs(mse - 0.25f) < 1e-5f);
    assert(std::abs(rmse - 0.5f) < 1e-5f);

    std::cout << "  ✅ Mean Squared Error Metrics Passed! (MSE=0.25, RMSE=0.5)" << std::endl;
}