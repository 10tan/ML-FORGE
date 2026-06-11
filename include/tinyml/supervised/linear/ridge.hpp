#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

namespace MLForge {
    namespace Supervised {
        namespace Linear {

            template <typename T>
            class RidgeRegression {
            private:
                T m_alpha;          // L2 Regularization strength
                T m_lr;             // Learning rate
                size_t m_epochs;    // Max training iterations
                std::vector<T> m_weights;
                T m_bias = T(0.0);
                bool m_is_fitted = false;

            public:
                RidgeRegression(T alpha = T(1.0), T learning_rate = T(0.01), size_t epochs = 1000)
                    : m_alpha(alpha), m_lr(learning_rate), m_epochs(epochs) {}

                /**
                 * @brief Trains the Ridge model using L2 regularized gradient descent
                 */
                void fit(const Core::Matrix<T>& X, const std::vector<T>& y) {
                    size_t num_samples = X.rows();
                    size_t num_features = X.cols();

                    if (num_samples != y.size() || num_samples == 0) {
                        throw std::runtime_error("Ridge Error: Matrix rows must match label vector size.");
                    }

                    // Initialize weights to zero
                    m_weights.assign(num_features, T(0.0));
                    m_bias = T(0.0);

                    // Gradient Descent Loop
                    for (size_t epoch = 0; epoch < m_epochs; ++epoch) {
                        std::vector<T> dw(num_features, T(0.0));
                        T db = T(0.0);

                        // Compute predictions and errors across the batch
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

                        // Apply standard averaging and add the L2 penalty gradient to weights
                        T scale = T(1.0) / static_cast<T>(num_samples);
                        for (size_t j = 0; j < num_features; ++j) {
                            // Derivative: (1/N)*dl/dw + alpha*w
                            dw[j] = (dw[j] * scale) + (m_alpha * m_weights[j]);
                            m_weights[j] -= m_lr * dw[j]; // Update weights
                        }
                        
                        db *= scale;
                        m_bias -= m_lr * db; // Update bias (typically left unregularized)
                    }

                    m_is_fitted = true;
                }

                /**
                 * @brief Predicts continuous outputs for an input matrix
                 */
                std::vector<T> predict(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("Ridge Error: Model must be fitted first.");

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