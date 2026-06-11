#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>

inline void test_preprocessing_and_io() {
    std::cout << "  ⚙️  Initializing Preprocessing Scalers & Disk I/O Tests..." << std::endl;

    std::vector<float> data = {10.0f, 20.0f, 30.0f};
    MLForge::Core::Matrix<float> X(3, 1, data);

    // 1. Validate MinMaxScaler (Bounds mapped precisely to [0, 1])
    MLForge::Utils::Preprocessing::MinMaxScaler<float> min_max_scaler;
    MLForge::Core::Matrix<float> X_min_max = min_max_scaler.fit_transform(X);

    assert(std::abs(X_min_max(0, 0) - 0.0f) < 1e-5f);
    assert(std::abs(X_min_max(1, 0) - 0.5f) < 1e-5f);
    assert(std::abs(X_min_max(2, 0) - 1.0f) < 1e-5f);

    // 2. Validate StandardScaler (Mean = 20, Population Std Dev = 8.164965)
    MLForge::Utils::Preprocessing::StandardScaler<float> std_scaler;
    MLForge::Core::Matrix<float> X_std = std_scaler.fit_transform(X);

    assert(std::abs(X_std(0, 0) - (-1.2247449f)) < 1e-4f);
    assert(std::abs(X_std(1, 0) - 0.0f) < 1e-4f);
    assert(std::abs(X_std(2, 0) - 1.2247449f) < 1e-4f);

    // 3. Validate Disk CSV Export
    std::string test_file = "test_output.csv";
    X_min_max.to_csv(test_file);

    // Verify file exists on disk
    std::ifstream check_stream(test_file);
    assert(check_stream.good());
    check_stream.close();
    std::remove(test_file.c_str()); // Clean workspace file system garbage

    std::cout << "  ✅ Preprocessing & I/O Assertions Cleared Successfully!" << std::endl;
}