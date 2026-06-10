#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>

inline void test_tensor_indexing_and_scalar() {
    std::cout << "  ⚙️  Initializing Multi-Dimensional Tensor Structural Test..." << std::endl;

    // 1. Instantiate a 3D Tensor of shape 2 x 3 x 4 (e.g., 2 batches, 3 rows, 4 columns)
    // Total elements = 2 * 3 * 4 = 24
    MLForge::Core::Tensor<float> T({2, 3, 4}, 0.0f);

    // Verify dimensions are stored correctly
    assert(T.rank() == 3);
    assert(T.size() == 24);
    assert(T.shape()[0] == 2);
    assert(T.shape()[1] == 3);
    assert(T.shape()[2] == 4);

    // 2. Assign values to specific high-dimensional coordinate slots
    T({0, 1, 2}) = 15.5f;
    T({1, 2, 3}) = 88.0f;

    // Verify coordinate extraction matches exactly
    assert(T({0, 1, 2}) == 15.5f);
    assert(T({1, 2, 3}) == 88.0f);
    assert(T({0, 0, 0}) == 0.0f); // Unassigned index should remain initial value

    // 3. Test Scalar Transformation Operator: T2 = T * 2.0
    MLForge::Core::Tensor<float> T2 = T * 2.0f;

    // Verify shapes match and transformation applied to modified coordinates
    assert(T2.size() == 24);
    assert(T2({0, 1, 2}) == 31.0f);
    assert(T2({1, 2, 3}) == 176.0f);

    // 4. Test Out-of-Bounds Error Handling
    bool caught_exception = false;
    try {
        // Row index 5 is out of bounds for shape {2, 3, 4}
        T({0, 5, 2}) = 1.0f;
    } catch (const std::out_of_range&) {
        caught_exception = true;
    }
    assert(caught_exception == true);

    std::cout << "  ✅ Tensor Indexing & Transformation Passed!" << std::endl;
}