#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace MLForge {
    namespace Supervised {
        namespace Linear {

            template <typename T>
            class LassoRegression {
            private:
                T m_alpha;          // L1 Regularization strength
                T m_lr;             // Learning rate
                size_t m_epochs;    // Max training iterations
                std::vector<T> m_weights;
                T m_bias = T(0.0);
                bool m_is_fitted = false;

                // Helper: Sign function for L1 derivative math
                T sign(T val) const {
                    if (val > T(0.0)) return T(1.0);
                    if (val < T(0.0)) return -T(1.0);
                    return T(0.0);
                }

            public:
                LassoRegression(T alpha = T(1.0), T learning_rate = T(0.01), size_t epochs = 1000)
                    : m_alpha(alpha), m_lr(learning_rate), m_epochs(epochs) {}

                /**
                 * @brief Trains the Lasso model using L1 regularized gradient updates
                 */
                void fit(const Core::Matrix<T>& X, const std::vector<T>& y) {
                    size_t num_samples = X.rows();
                    size_t num_features = X.cols();

                    if (num_samples != y.size() || num_samples == 0) {
                        throw std::runtime_error("Lasso Error: Matrix rows must match label vector size.");
                    }

                    m_weights.assign(num_features, T(0.0));
                    m_bias = T(0.0);

                    T scale = T(1.0) / static_cast<T>(num_samples);

                    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
                        std::vector<T> dw(num_features, T(0.0));
                        T db = T(0.0);

                        // 1. Compute baseline error gradients
                        for (size_t i = 0; i < num_samples; ++i) {
                            T pred = m_bias;
                            for (size_t j = 0; j < num_features; ++j) {
                                pred += X(i, j) * m_weights[j];
                            }
                            
                            T error = pred - y[i];
                            
                            for (size_t j = 0; j < num_features; ++j) {
                                dw[j] += error * X(i, j);
                            }
                            db += error;
                        }

                        // 2. Apply updates combined with constant L1 sign penalty forces
                        for (size_t j = 0; j < num_features; ++j) {
                            dw[j] = (dw[j] * scale) + (m_alpha * sign(m_weights[j]));
                            m_weights[j] -= m_lr * dw[j];
                            
                            // Proximal subgradient check: if a weight crosses 0 due to penalty, clip it to 0
                            // This ensures the sparse feature property of Lasso holds perfectly
                            if (std::abs(m_weights[j]) < T(1e-4)) {
                                m_weights[j] = T(0.0);
                            }
                        }
                        
                        m_bias -= m_lr * (db * scale);
                    }

                    m_is_fitted = true;
                }

                std::vector<T> predict(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("Lasso Error: Model must be fitted first.");

                    size_t num_samples = X.rows();
                    size_t num_features = X.cols();
                    std::vector<T> predictions(num_samples);

                    for (size_t i = 0; i < num_samples; ++i) {
                        T pred = m_bias;
                        for (size_t j = 0; j < num_features; ++j) {
                            pred += X(i, j) * m_weights[j];
                        }
                        predictions[i] = pred;
                    }

                    return predictions;
                }

                const std::vector<T>& get_weights() const { return m_weights; }
                T get_bias() const { return m_bias; }
            };

        }
    }
}