#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace MLForge {
    namespace Utils {
        namespace Preprocessing {

            // ============================================================================
            // 🧠 StandardScaler: Z-score Normalization (Mean=0, Std=1)
            // ============================================================================
            template <typename T>
            class StandardScaler {
            private:
                std::vector<T> m_means;
                std::vector<T> m_stds;
                bool m_is_fitted = false;

            public:
                StandardScaler() = default;

                /**
                 * @brief Calculates running column-wise mean and standard deviation matrices.
                 */
                void fit(const Core::Matrix<T>& X) {
                    size_t rows = X.rows();
                    size_t cols = X.cols();
                    if (rows == 0) throw std::runtime_error("Scaler Error: Cannot fit an empty matrix.");

                    m_means.assign(cols, T(0));
                    m_stds.assign(cols, T(0));

                    // 1. Calculate running mean column aggregates
                    for (size_t c = 0; c < cols; ++c) {
                        T sum = 0;
                        for (size_t r = 0; r < rows; ++r) sum += X(r, c);
                        m_means[c] = sum / static_cast<T>(rows);
                    }

                    // 2. Calculate column standard deviations
                    for (size_t c = 0; c < cols; ++c) {
                        T variance_sum = 0;
                        for (size_t r = 0; r < rows; ++r) {
                            T diff = X(r, c) - m_means[c];
                            variance_sum += diff * diff;
                        }
                        // 1e-7 epsilon protects from division-by-zero on completely flat data columns
                        m_stds[c] = std::sqrt(variance_sum / static_cast<T>(rows)) + T(1e-7);
                    }
                    m_is_fitted = true;
                }

                /**
                 * @brief Transforms data based on pre-calculated mean and standard deviation states.
                 */
                Core::Matrix<T> transform(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("Scaler Error: Model must be fitted before transforming.");
                    
                    Core::Matrix<T> X_scaled(X.rows(), X.cols());
                    for (size_t c = 0; c < X.cols(); ++c) {
                        for (size_t r = 0; r < X.rows(); ++r) {
                            X_scaled(r, c) = (X(r, c) - m_means[c]) / m_stds[c];
                        }
                    }
                    return X_scaled;
                }

                Core::Matrix<T> fit_transform(const Core::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                const std::vector<T>& means() const { return m_means; }
                const std::vector<T>& stds() const { return m_stds; }
            };


            // ============================================================================
            // 🧠 MinMaxScaler: Range Rescaling (Bounded between 0.0 and 1.0)
            // ============================================================================
            template <typename T>
            class MinMaxScaler {
            private:
                std::vector<T> m_mins;
                std::vector<T> m_maxs;
                bool m_is_fitted = false;

            public:
                MinMaxScaler() = default;

                /**
                 * @brief Detects absolute minimum and maximum boundary coordinates for feature columns.
                 */
                void fit(const Core::Matrix<T>& X) {
                    size_t rows = X.rows();
                    size_t cols = X.cols();
                    if (rows == 0) throw std::runtime_error("Scaler Error: Cannot fit an empty matrix.");

                    m_mins.assign(cols, T(0));
                    m_maxs.assign(cols, T(0));

                    for (size_t c = 0; c < cols; ++c) {
                        T min_val = X(0, c);
                        T max_val = X(0, c);
                        for (size_t r = 1; r < rows; ++r) {
                            if (X(r, c) < min_val) min_val = X(r, c);
                            if (X(r, c) > max_val) max_val = X(r, c);
                        }
                        m_mins[c] = min_val;
                        m_maxs[c] = max_val;
                    }
                    m_is_fitted = true;
                }

                /**
                 * @brief Compresses inputs into an exact [0, 1] scale interval.
                 */
                Core::Matrix<T> transform(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("Scaler Error: Model must be fitted before transforming.");
                    
                    Core::Matrix<T> X_scaled(X.rows(), X.cols());
                    for (size_t c = 0; c < X.cols(); ++c) {
                        T range = m_maxs[c] - m_mins[c] + T(1e-7); // Epsilon scaling protection
                        for (size_t r = 0; r < X.rows(); ++r) {
                            X_scaled(r, c) = (X(r, c) - m_mins[c]) / range;
                        }
                    }
                    return X_scaled;
                }

                Core::Matrix<T> fit_transform(const Core::Matrix<T>& X) {
                    fit(X);
                    return transform(X);
                }

                const std::vector<T>& mins() const { return m_mins; }
                const std::vector<T>& maxs() const { return m_maxs; }
            };

        }
    }
}