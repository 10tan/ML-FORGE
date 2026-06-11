#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace MLForge {
    namespace Utils {
        namespace Visualization {

            template <typename T>
            class DataVisualizer {
            private:
                // ANSI Terminal Color Codes Mapping
                static std::string get_ansi_color(size_t color_code) {
                    switch(color_code % 6) {
                        case 0: return "\033[1;31m"; // Bold Red
                        case 1: return "\033[1;34m"; // Bold Blue
                        case 2: return "\033[1;32m"; // Bold Green
                        case 3: return "\033[1;33m"; // Bold Yellow
                        case 4: return "\033[1;35m"; // Bold Magenta
                        case 5: return "\033[1;36m"; // Bold Cyan
                        default: return "\033[0m";
                    }
                }

                static std::string reset_ansi() {
                    return "\033[0m";
                }

            public:
                DataVisualizer() = default;

                /**
                 * @brief Renders a colorized 2D scatter boundary plot directly to console stdout.
                 */
                static void scatter_plot(const Core::Matrix<T>& X, const Core::Matrix<T>& y, 
                                         const std::vector<size_t>& labels = {},
                                         size_t width = 60, size_t height = 20,
                                         const std::string& save_path = "") {
                    if (X.rows() != y.rows() || X.rows() == 0) {
                        std::cerr << "Visualization Error: Dimensions must match." << std::endl;
                        return;
                    }

                    T min_x = X(0, 0), max_x = X(0, 0);
                    T min_y = y(0, 0), max_y = y(0, 0);
                    for (size_t i = 0; i < X.rows(); ++i) {
                        if (X(i, 0) < min_x) min_x = X(i, 0);
                        if (X(i, 0) > max_x) max_x = X(i, 0);
                        if (y(i, 0) < min_y) min_y = y(i, 0);
                        if (y(i, 0) > max_y) max_y = y(i, 0);
                    }

                    if (std::abs(max_x - min_x) < T(1e-5)) max_x += T(1);
                    if (std::abs(max_y - min_y) < T(1e-5)) max_y += T(1);

                    std::vector<std::string> canvas(height, std::string(width, ' '));
                    std::vector<std::vector<int>> label_grid(height, std::vector<int>(width, -1));

                    for (size_t i = 0; i < X.rows(); ++i) {
                        T pct_x = (X(i, 0) - min_x) / (max_x - min_x);
                        T pct_y = (y(i, 0) - min_y) / (max_y - min_y);

                        size_t col = static_cast<size_t>(pct_x * static_cast<T>(width - 1));
                        size_t row = (height - 1) - static_cast<size_t>(pct_y * static_cast<T>(height - 1));

                        if (row < height && col < width) {
                            canvas[row][col] = '*';
                            if (!labels.empty() && i < labels.size()) {
                                label_grid[row][col] = static_cast<int>(labels[i]);
                            }
                        }
                    }

                    std::cout << "\n--- MLForge Colorized 2D Scatter Plot ---" << std::endl;
                    for (size_t r = 0; r < height; ++r) {
                        std::cout << "| ";
                        for (size_t c = 0; c < width; ++c) {
                            if (canvas[r][c] == '*') {
                                if (label_grid[r][c] != -1) {
                                    std::cout << get_ansi_color(label_grid[r][c]) << "*" << reset_ansi();
                                } else {
                                    std::cout << "*";
                                }
                            } else {
                                std::cout << ' ';
                            }
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "+" << std::string(width, '-') << std::endl;
                    std::cout << " Min X: " << min_x << std::string(width > 16 ? width - 16 : 4, ' ') << "Max X: " << max_x << "\n" << std::endl;

                    if (!save_path.empty()) {
                        std::ofstream file(save_path);
                        if (file.is_open()) {
                            file << "\n--- MLForge Text-Based 2D Scatter Plot ---\n";
                            for (size_t r = 0; r < height; ++r) file << "| " << canvas[r] << "\n";
                            file << "+" << std::string(width, '-') << "\n";
                            file << " Min X: " << min_x << "  Max X: " << max_x << "\n";
                            file.close();
                        }
                    }
                }

                /**
                 * @brief Generates frequency distribution bars for specific target matrix features.
                 */
                static void histogram(const Core::Matrix<T>& matrix, size_t col_idx, 
                                      size_t bins = 10, size_t max_bar_width = 40,
                                      bool colorize = true,
                                      const std::string& save_path = "") {
                    if (matrix.rows() == 0 || col_idx >= matrix.cols()) return;

                    T min_val = matrix(0, col_idx);
                    T max_val = matrix(0, col_idx);
                    for (size_t r = 1; r < matrix.rows(); ++r) {
                        if (matrix(r, col_idx) < min_val) min_val = matrix(r, col_idx);
                        if (matrix(r, col_idx) > max_val) max_val = matrix(r, col_idx);
                    }

                    T range = max_val - min_val;
                    if (std::abs(range) < T(1e-5)) range = T(1);

                    std::vector<size_t> frequencies(bins, 0);
                    for (size_t r = 0; r < matrix.rows(); ++r) {
                        T val = matrix(r, col_idx);
                        size_t bin_idx = static_cast<size_t>(((val - min_val) / range) * static_cast<T>(bins));
                        if (bin_idx >= bins) bin_idx = bins - 1;
                        frequencies[bin_idx]++;
                    }

                    size_t max_freq = *std::max_element(frequencies.begin(), frequencies.end());
                    if (max_freq == 0) max_freq = 1;

                    std::cout << "\n--- MLForge Distribution Frequency Histogram ---" << std::endl;
                    std::stringstream file_ss;
                    file_ss << "\n--- MLForge Distribution Frequency Histogram ---\n";
                    T bin_width = range / static_cast<T>(bins);

                    for (size_t b = 0; b < bins; ++b) {
                        T current_bin_start = min_val + (static_cast<T>(b) * bin_width);
                        size_t bar_length = (frequencies[b] * max_bar_width) / max_freq;
                        std::string bar = std::string(bar_length, '#'); 

                        std::cout << " [" << std::setw(6) << std::fixed << std::setprecision(2) << current_bin_start << "] | ";
                        if (colorize) {
                            std::cout << get_ansi_color(b) << bar << reset_ansi() << " (" << frequencies[b] << ")" << std::endl;
                        } else {
                            std::cout << bar << " (" << frequencies[b] << ")" << std::endl;
                        }

                        file_ss << " [" << std::setw(6) << std::fixed << std::setprecision(2) << current_bin_start 
                                << "] | " << bar << " (" << frequencies[b] << ")\n";
                    }
                    std::cout << "--------------------------------------------------\n" << std::endl;
                    file_ss << "--------------------------------------------------\n";

                    if (!save_path.empty()) {
                        std::ofstream file(save_path);
                        if (file.is_open()) {
                            file << file_ss.str();
                            file.close();
                        }
                    }
                }

                /**
                 * @brief Evaluates classification metric counts in an aligned layout matrix block.
                 */
                static void confusion_matrix(const std::vector<size_t>& y_true, 
                                             const std::vector<size_t>& y_pred,
                                             const std::string& save_path = "") {
                    if (y_true.size() != y_pred.size() || y_true.empty()) {
                        std::cerr << "Visualization Error: Matrix label vector dimensions must match." << std::endl;
                        return;
                    }

                    size_t num_classes = 0;
                    for (size_t i = 0; i < y_true.size(); ++i) {
                        if (y_true[i] > num_classes) num_classes = y_true[i];
                        if (y_pred[i] > num_classes) num_classes = y_pred[i];
                    }
                    num_classes += 1; 

                    std::vector<std::vector<size_t>> matrix(num_classes, std::vector<size_t>(num_classes, 0));
                    for (size_t i = 0; i < y_true.size(); ++i) {
                        matrix[y_true[i]][y_pred[i]]++;
                    }

                    std::stringstream ss;
                    ss << "\n📋 --- MLForge Classification Confusion Matrix ---" << std::endl;
                    ss << "          Predicted Class Layout" << std::endl;
                    ss << "          ";
                    for (size_t c = 0; c < num_classes; ++c) ss << std::setw(6) << "[" << c << "]";
                    ss << std::endl;
                    ss << "          " << std::string(num_classes * 8, '-') << std::endl;

                    for (size_t r = 0; r < num_classes; ++r) {
                        if (r == 0) ss << "True [" << r << "] | ";
                        else        ss << "     [" << r << "] | ";
                        
                        for (size_t c = 0; c < num_classes; ++c) {
                            ss << std::setw(7) << matrix[r][c];
                        }
                        ss << std::endl;
                    }
                    ss << "          " << std::string(num_classes * 8, '-') << "\n" << std::endl;

                    std::cout << ss.str();

                    if (!save_path.empty()) {
                        std::ofstream file(save_path);
                        if (file.is_open()) {
                            file << ss.str();
                            file.close();
                        }
                    }
                }
            };

        }
    }
}