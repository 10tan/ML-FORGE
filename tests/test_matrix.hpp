#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>

inline void test_matrix_addition() {
    MLForge::Core::Matrix<float> A(2, 2, 2.0f);
    MLForge::Core::Matrix<float> B(2, 2, 3.0f);
    
    MLForge::Core::Matrix<float> C = A + B;
    
    // Validate every element equals 5.0
    for(size_t i = 0; i < C.size(); ++i) {
        assert(std::abs(C.data()[i] - 5.0f) < 1e-5);
    }
    std::cout << "  ⚙️  Initializing Matrix Test..." << std::endl;
    std::cout << "  ✅ Matrix Addition Passed!" << std::endl;
}