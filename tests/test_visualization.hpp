#pragma once
#include "mlforge.hpp"
#include <vector>
#include <fstream>
#include <cassert>

inline void test_terminal_and_disk_plots() {
    std::cout << "  ⚙️  Initializing Canvas Render Graphics & File I/O Sandbox..." << std::endl;

    // Linear Trend Dataset
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    MLForge::Core::Matrix<float> X(10, 1, x_data);
    std::vector<float> y_data = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f};
    MLForge::Core::Matrix<float> y(10, 1, y_data);

    // Render to terminal AND save copy out to workspace file system disk sheets
    std::string scatter_file = "test_scatter.txt";
    MLForge::Utils::Visualization::DataVisualizer<float>::scatter_plot(X, y, 40, 10, scatter_file);

    // Verify scatter file output exists
    std::ifstream f1(scatter_file);
    assert(f1.good());
    f1.close();
    std::remove(scatter_file.c_str());

    std::cout << "  ✅ Terminal Rendering and Graphics Export Checked out Successfully!" << std::endl;
}