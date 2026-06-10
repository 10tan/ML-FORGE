// src/supervised/linear/linear_class.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
        LogisticRegression<T>::LogisticRegression(T learning_rate, size_t iterations)
            : m_bias(0), m_lr(learning_rate), m_iters(iterations) {}

        template <typename T>
        void LogisticRegression<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
            size_t m = X.rows();
            size_t n = X.cols();

            m_weights = Core::Matrix<T>(n, 1, T(0));
            m_bias = T(0);

            for (size_t iter = 0; iter < m_iters; ++iter) {
                // 1. Compute raw linear prediction: Z = X * W + b
                Core::Matrix<T> Z = X.matmul(m_weights);
                for (size_t i = 0; i < m; ++i) {
                    Z(i, 0) += m_bias;
                }

                // 2. Squash to probabilities: y_hat = sigmoid(Z)
                Core::Matrix<T> y_hat = NN::Activations::Sigmoid<T>::forward(Z);

                // 3. Compute error (y_hat - y)
                Core::Matrix<T> error = y_hat - y;

                // 4. Gradient updates: dW = (1/m) * X^T * error
                Core::Matrix<T> XT = X.transpose();
                Core::Matrix<T> dW = XT.matmul(error) * (T(1) / static_cast<T>(m));

                T db = 0;
                for (size_t i = 0; i < m; ++i) {
                    db += error(i, 0);
                }
                db /= static_cast<T>(m);

                // 5. Adjust optimization vectors
                m_weights = m_weights - (dW * m_lr);
                m_bias -= m_lr * db;
            }
        }

        template <typename T>
        Core::Matrix<T> LogisticRegression<T>::predict_proba(const Core::Matrix<T>& X) const {
            size_t m = X.rows();
            Core::Matrix<T> Z = X.matmul(m_weights);
            for (size_t i = 0; i < m; ++i) {
                Z(i, 0) += m_bias;
            }
            return NN::Activations::Sigmoid<T>::forward(Z);
        }

        template <typename T>
        Core::Matrix<T> LogisticRegression<T>::predict(const Core::Matrix<T>& X) const {
            Core::Matrix<T> probas = predict_proba(X);
            Core::Matrix<T> classes(probas.rows(), probas.cols());
            
            // Use the standard matrix indexing operators to update values cleanly
            for (size_t r = 0; r < probas.rows(); ++r) {
                for (size_t c = 0; c < probas.cols(); ++c) {
                    classes(r, c) = (probas(r, c) >= 0.5f) ? T(1) : T(0);
                }
            }
            return classes;
        }

    }
}