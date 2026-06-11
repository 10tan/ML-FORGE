#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace MLForge {
    namespace Supervised {
        namespace Linear {

            template <typename T>
            class ElasticNetRegression {
            private:
                T m_alpha;     // Overall regularization penalty multiplier
                T m_l1_ratio;  // Mixing parameter rho (1.0 = Lasso, 0.0 = Ridge)
                T m_lr;        // Learning rate
                size_t m_epochs;
                std::vector<T> m_weights;
                T m_bias = T(0.0);
                bool m_is_fitted = false;

                T sign(T val) const {
                    if (val > T(0.0)) return T(1.0);
                    if (val < T(0.0)) return -T(1.0);
                    return T(0.0);
                }

            public:
                ElasticNetRegression(T alpha = T(1.0), T l1_ratio = T(0.5), T learning_rate = T(0.01), size_t epochs = 1000)
                    : m_alpha(alpha), m_l1_ratio(l1_ratio), m_lr(learning_rate), m_epochs(epochs) {}

                /**
                 * @brief Trains the model using a blend of L1 and L2 regularizations
                 */
                void fit(const Core::Matrix<T>& X, const std::vector<T>& y) {
                    size_t num_samples = X.rows();
                    size_t num_features = X.cols();

                    if (num_samples != y.size() || num_samples == 0) {
                        throw std::runtime_error("ElasticNet Error: Matrix row and target vector size boundaries mismatch.");
                    }

                    m_weights.assign(num_features, T(0.0));
                    m_bias = T(0.0);

                    T scale = T(1.0) / static_cast<T>(num_samples);

                    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
                        std::vector<T> dw(num_features, T(0.0));
                        T db = T(0.0);

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

                        // Apply the blended L1/L2 penalties to the feature gradients
                        for (size_t j = 0; j < num_features; ++j) {
                            T l1_penalty_grad = m_alpha * m_l1_ratio * sign(m_weights[j]);
                            T l2_penalty_grad = m_alpha * (T(1.0) - m_l1_ratio) * m_weights[j];
                            
                            dw[j] = (dw[j] * scale) + l1_penalty_grad + l2_penalty_grad;
                            m_weights[j] -= m_lr * dw[j];

                            // Proximal clipping for feature elimination stability
                            if (m_l1_ratio > T(0.0) && std::abs(m_weights[j]) < T(1e-4)) {
                                m_weights[j] = T(0.0);
                            }
                        }

                        m_bias -= m_lr * (db * scale);
                    }

                    m_is_fitted = true;
                }

                std::vector<T> predict(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("ElasticNet Error: Model must be fitted first.");
                    size_t num_samples = X.rows();
                    std::vector<T> predictions(num_samples);
                    for (size_t i = 0; i < num_samples; ++i) {
                        T pred = m_bias;
                        for (size_t j = 0; j < X.cols(); ++j) {
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