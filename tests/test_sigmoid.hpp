#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>

inline void test_sigmoid_activation() {
    std::cout << "  ⚙️  Initializing Sigmoid Activation Value Test..." << std::endl;

    // Standard boundary checks: 
    // Sigmoid(0) = 0.5
    // Sigmoid(large positive) -> near 1.0
    // Sigmoid(large negative) -> near 0.0
    std::vector<float> data = {0.0f, -100.0f, 100.0f};
    MLForge::Core::Matrix<float> X(1, 3, data);

    MLForge::Core::Matrix<float> out = MLForge::Supervised::NN::Activations::Sigmoid<float>::forward(X);

    assert(std::abs(out(0, 0) - 0.5f) < 1e-5f);
    assert(out(0, 1) < 0.01f);
    assert(out(0, 2) > 0.99f);

    std::cout << "  ✅ Sigmoid Activation Passed!" << std::endl;
}