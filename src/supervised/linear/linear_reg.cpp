// src/supervised/linear/linear_reg.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
        LinearRegression<T>::LinearRegression(T learning_rate, size_t iterations)
            : m_bias(0), m_lr(learning_rate), m_iters(iterations) {}

        template <typename T>
        void LinearRegression<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
            size_t m = X.rows();
            size_t n = X.cols();

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

                // 2. Compute Error: error = y_hat - y
                Core::Matrix<T> error = y_hat - y;

                // 3. Compute Gradients: dW = (X^T * error) * (1 / m)
                Core::Matrix<T> XT = X.transpose();
                Core::Matrix<T> dW = XT.matmul(error) * (T(1) / static_cast<T>(m));

                // Compute Bias Gradient: db = sum(error) * (1 / m)
                T db = 0;
                for (size_t i = 0; i < m; ++i) {
                    db += error(i, 0);
                }
                db /= static_cast<T>(m);

                // 4. Update Parameters: W = W - (lr * dW)
                m_weights = m_weights - (dW * m_lr);
                m_bias -= m_lr * db;
            }
        }

        template <typename T>
        Core::Matrix<T> LinearRegression<T>::predict(const Core::Matrix<T>& X) const {
            size_t m = X.rows();
            Core::Matrix<T> predictions = X.matmul(m_weights);
            for (size_t i = 0; i < m; ++i) {
                predictions(i, 0) += m_bias;
            }
            return predictions;
        }

    }
}