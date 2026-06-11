// src/utils/preprocessing/scaler.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Utils {
        namespace Preprocessing {

            // ============================================================================
            // StandardScaler Implementation
            // ============================================================================
            template <typename T>
            void StandardScaler<T>::fit(const Core::Matrix<T>& X) {
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

            template <typename T>
            Core::Matrix<T> StandardScaler<T>::transform(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) throw std::runtime_error("Scaler Error: Model must be fitted before transforming.");
                
                Core::Matrix<T> X_scaled(X.rows(), X.cols());
                for (size_t c = 0; c < X.cols(); ++c) {
                    for (size_t r = 0; r < X.rows(); ++r) {
                        X_scaled(r, c) = (X(r, c) - m_means[c]) / m_stds[c];
                    }
                }
                return X_scaled;
            }

            template <typename T>
            Core::Matrix<T> StandardScaler<T>::fit_transform(const Core::Matrix<T>& X) {
                fit(X);
                return transform(X);
            }

            // ============================================================================
            // MinMaxScaler Implementation
            // ============================================================================
            template <typename T>
            void MinMaxScaler<T>::fit(const Core::Matrix<T>& X) {
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

            template <typename T>
            Core::Matrix<T> MinMaxScaler<T>::transform(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) throw std::runtime_error("Scaler Error: Model must be fitted before transforming.");
                
                Core::Matrix<T> X_scaled(X.rows(), X.cols());
                for (size_t c = 0; c < X.cols(); ++c) {
                    T range = m_maxs[c] - m_mins[c] + T(1e-7); // Epsilon protection
                    for (size_t r = 0; r < X.rows(); ++r) {
                        X_scaled(r, c) = (X(r, c) - m_mins[c]) / range;
                    }
                }
                return X_scaled;
            }

            template <typename T>
            Core::Matrix<T> MinMaxScaler<T>::fit_transform(const Core::Matrix<T>& X) {
                fit(X);
                return transform(X);
            }

        }
    }
}