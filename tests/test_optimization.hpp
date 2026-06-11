#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>
#include <iostream>

inline void test_optimization_and_utils() {
    std::cout << "  ⚙️  Initializing Partition Split & Separated Optimizer Tests..." << std::endl;

    // 1. Verify train_test_split ratios
    std::vector<float> features(20, 1.0f); // 10 rows, 2 columns dataset
    MLForge::Core::Matrix<float> X(10, 2, features);
    std::vector<float> targets(10, 0.0f);
    MLForge::Core::Matrix<float> y(10, 1, targets);

    MLForge::Core::Matrix<float> X_train, X_test, y_train, y_test;
    MLForge::Utils::train_test_split(X, y, X_train, X_test, y_train, y_test, 0.40, 42);

    assert(X_train.rows() == 6);
    assert(X_test.rows() == 4);

    // 2. Verify Standalone SGDMomentum Weight Update Steps
    MLForge::Core::Matrix<float> w_sgd(1, 1, std::vector<float>{5.0f});
    MLForge::Core::Matrix<float> b_sgd(1, 1, std::vector<float>{2.0f});
    MLForge::Core::Matrix<float> dW_sgd(1, 1, std::vector<float>{1.0f});
    MLForge::Core::Matrix<float> db_sgd(1, 1, std::vector<float>{0.5f});

    MLForge::Supervised::NN::Optimizers::SGDMomentum<float> sgd_opt(0.1f, 0.9f);
    sgd_opt.update(w_sgd, b_sgd, dW_sgd, db_sgd);

    // First step: velocity = 0.1 * 1.0 = 0.1. w = 5.0 - 0.1 = 4.9f
    assert(std::abs(w_sgd(0, 0) - 4.9f) < 1e-5f);

    // 3. Verify Standalone Adam Weight Update Steps
    MLForge::Core::Matrix<float> w_adam(1, 1, std::vector<float>{10.0f});
    MLForge::Core::Matrix<float> b_adam(1, 1, std::vector<float>{5.0f});
    MLForge::Core::Matrix<float> dW_adam(1, 1, std::vector<float>{2.0f}); 
    MLForge::Core::Matrix<float> db_adam(1, 1, std::vector<float>{1.0f});

    MLForge::Supervised::NN::Optimizers::Adam<float> adam_opt(0.1f);
    adam_opt.update(w_adam, b_adam, dW_adam, db_adam);

    // Parameter states must step downwards dynamically
    assert(w_adam(0, 0) < 10.0f);
    assert(b_adam(0, 0) < 5.0f);

    std::cout << "  ✅ Split Utilities, SGD Momentum, and Adam Optimizers Verified!" << std::endl;
}