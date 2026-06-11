#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>

inline void test_relu_forward() {
    std::cout << "  ⚙️  Initializing ReLU Activation Test..." << std::endl;

    // Mixed matrix containing negative, zero, and positive values
    std::vector<float> values = {-2.5f, 0.0f, 4.2f, -0.1f};
    MLForge::Core::Matrix<float> X(2, 2, values);

    // Run forward pass
    MLForge::Supervised::NN::Activations::ReLU<float> relu;
    MLForge::Core::Matrix<float> activated = relu.forward(X);

    // Check boundary thresholding max(0, x)
    assert(activated(0, 0) == 0.0f);
    assert(activated(0, 1) == 0.0f);
    assert(activated(1, 0) == 4.2f);
    assert(activated(1, 1) == 0.0f);

    // Upstream gradient simulate
    std::vector<float> grad_values = {1.0f, 1.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> upstream_grad(2, 2, grad_values);
    
    // Run backward pass
    MLForge::Core::Matrix<float> downstream_grad = relu.backward(X, upstream_grad);

    // Gradient should be 1.0 only where input was > 0
    assert(downstream_grad(0, 0) == 0.0f);
    assert(downstream_grad(0, 1) == 0.0f);
    assert(downstream_grad(1, 0) == 1.0f);
    assert(downstream_grad(1, 1) == 0.0f);

    std::cout << "  ✅ ReLU Activation Passed! (Rectification and gradient functional)" << std::endl;
}