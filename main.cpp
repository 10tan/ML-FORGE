#include <iostream>
#include "include/mlforge.hpp"

int main() {
    MLForge::system_check();

    std::cout << "--- Testing Matrix directly from Root ---\n" << std::endl;

    // Create a simple 2x3 matrix filled with 5.5
    MLForge::Core::Matrix<float> mat(2, 3);
    
    // Assign values manually using your flat index operator
    mat(0, 0) = 1.1f; mat(0, 1) = 2.2f; mat(0, 2) = 3.3f;
    mat(1, 0) = 4.4f; mat(1, 1) = 5.5f; mat(1, 2) = 6.6f;

    std::cout << "Matrix Output:" << std::endl;
    mat.print();

    return 0;
}