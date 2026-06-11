#pragma once
#include "tinyml/core/matrix.hpp"
#include "tinyml/supervised/base.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class LogisticRegression : public BasePredictor<T> { // 🎯 FIX: Change this to LogisticRegression
        private:
            Core::Matrix<T> m_weights;
            T m_bias;
            T m_lr;
            size_t m_iters;
        public:
            LogisticRegression(T learning_rate = T(0.01), size_t iterations = 1000)
                : m_bias(T(0.0)), m_lr(learning_rate), m_iters(iterations) {}

            /**
             * @brief Fits a linear hyperplane through batch gradient descent optimization.
             */
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override {
                size_t m = X.rows();
                size_t n = X.cols();

                if (m == 0 || m != y.rows()) {
                    throw std::runtime_error("Linear Regression Error: Dimensional alignment mismatch between X and y.");
                }

                // Initialize weights matrix as an n x 1 column vector of zeros
                m_weights = Core::Matrix<T>(n, 1, T(0));
                m_bias = T(0);

                // Gradient Descent Optimization Loop
                for (size_t iter = 0; iter < m_iters; ++iter) {
                    // 1. Forward Pass: y_hat = (X * W) + bias
                    Core::Matrix<T> y_hat = X.matmul(m_weights);
                    for (size_t i = 0; i < m; ++i) {
                        y_hat(i, 0) += m_bias;
                    }

                    // 2. Compute Error Elements: error = y_hat - y
                    Core::Matrix<T> error(m, 1);
                    for (size_t i = 0; i < m; ++i) {
                        error(i, 0) = y_hat(i, 0) - y(i, 0);
                    }

                    // 3. Compute Gradients: dW = (X^T * error) * (1 / m)
                    Core::Matrix<T> XT = X.transpose();
                    Core::Matrix<T> dW_raw = XT.matmul(error);
                    
                    // Scale weight gradients manually to prevent matrix operator dependency issues
                    Core::Matrix<T> dW(n, 1);
                    for (size_t i = 0; i < n; ++i) {
                        dW(i, 0) = dW_raw(i, 0) / static_cast<T>(m);
                    }

                    // Compute Bias Gradient: db = sum(error) * (1 / m)
                    T db = 0;
                    for (size_t i = 0; i < m; ++i) {
                        db += error(i, 0);
                    }
                    db /= static_cast<T>(m);

                    // 4. Update Parameters: W = W - (lr * dW), b = b - (lr * db)
                    for (size_t i = 0; i < n; ++i) {
                        m_weights(i, 0) -= m_lr * dW(i, 0);
                    }
                    m_bias -= m_lr * db;
                }
            }

            /**
             * @brief Generates predicted target values using the optimized plane equations.
             */
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override {
                if (m_weights.rows() == 0) {
                    throw std::runtime_error("Linear Regression Error: Cannot predict before model fit loop execution.");
                }
                
                size_t m = X.rows();
                Core::Matrix<T> predictions = X.matmul(m_weights);
                for (size_t i = 0; i < m; ++i) {
                    predictions(i, 0) += m_bias;
                }
                return predictions;
            }

            const Core::Matrix<T>& weights() const { return m_weights; }
            T bias() const { return m_bias; }
        };

    }
}