#pragma once
#include "tinyml/core/matrix.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace MLForge {
    namespace Utils {
        namespace Visualization {

            template <typename T>
            class DataVisualizer {
            public:
                // Renders a 2D Scatter Plot of X vs Y to terminal and optionally saves it as a .txt file
                static void scatter_plot(const Core::Matrix<T>& X, const Core::Matrix<T>& y, 
                                         size_t width = 50, size_t height = 15,
                                         const std::string& save_path = "");

                // Renders an ASCII Frequency Histogram to terminal and optionally saves it as a .txt file
                static void histogram(const Core::Matrix<T>& matrix, size_t col_idx = 0, 
                                      size_t bins = 10, size_t max_bar_width = 40,
                                      const std::string& save_path = "");
            };

        }
    }
}