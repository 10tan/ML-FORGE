// src/utils/visualization/plot.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Utils {
        namespace Visualization {

            template <typename T>
            void DataVisualizer<T>::scatter_plot(const Core::Matrix<T>& X, const Core::Matrix<T>& y, 
                                                 size_t width, size_t height,
                                                 const std::string& save_path) {
                if (X.rows() != y.rows() || X.rows() == 0) {
                    std::cerr << "Visualization Error: Matrix dimensions must match and not be empty." << std::endl;
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

                for (size_t i = 0; i < X.rows(); ++i) {
                    T pct_x = (X(i, 0) - min_x) / (max_x - min_x);
                    T pct_y = (y(i, 0) - min_y) / (max_y - min_y);

                    size_t col = static_cast<size_t>(pct_x * static_cast<T>(width - 1));
                    size_t row = (height - 1) - static_cast<size_t>(pct_y * static_cast<T>(height - 1));

                    if (row < height && col < width) {
                        canvas[row][col] = '*';
                    }
                }

                std::stringstream ss;
                ss << "\n--- MLForge Text-Based 2D Scatter Plot ---" << std::endl;
                for (size_t r = 0; r < height; ++r) {
                    ss << "| " << canvas[r] << std::endl;
                }
                // FIXED: Standard ASCII string replacement to remove multi-character warnings
                ss << "+" << std::string(width, '-') << std::endl;
                ss << " Min X: " << min_x << std::string(width > 16 ? width - 16 : 4, ' ') << "Max X: " << max_x << "\n" << std::endl;

                std::cout << ss.str();

                if (!save_path.empty()) {
                    std::ofstream file(save_path);
                    if (file.is_open()) {
                        file << ss.str();
                        file.close();
                    }
                }
            }

            template <typename T>
            void DataVisualizer<T>::histogram(const Core::Matrix<T>& matrix, size_t col_idx, 
                                              size_t bins, size_t max_bar_width,
                                              const std::string& save_path) {
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

                std::stringstream ss;
                ss << "\n--- MLForge Distribution Frequency Histogram ---" << std::endl;
                T bin_width = range / static_cast<T>(bins);

                for (size_t b = 0; b < bins; ++b) {
                    T current_bin_start = min_val + (static_cast<T>(b) * bin_width);
                    size_t bar_length = (frequencies[b] * max_bar_width) / max_freq;
                    std::string bar = std::string(bar_length, '#'); 

                    // FIXED: Re-aligned closing bracket safety string
                    ss << " [" << std::setw(6) << std::fixed << std::setprecision(2) << current_bin_start 
                       << "] | " << bar << " (" << frequencies[b] << ")" << std::endl;
                }
                ss << "--------------------------------------------------\n" << std::endl;

                std::cout << ss.str();

                if (!save_path.empty()) {
                    std::ofstream file(save_path);
                    if (file.is_open()) {
                        file << ss.str();
                        file.close();
                    }
                }
            }

        }
    }
}