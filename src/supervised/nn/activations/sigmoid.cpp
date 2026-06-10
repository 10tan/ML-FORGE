// src/supervised/nn/activations/sigmoid.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                // Forward Pass: Maps numbers to a clean (0, 1) probability range
                template <typename T>
                Core::Matrix<T> Sigmoid<T>::forward(const Core::Matrix<T>& x) {
                    Core::Matrix<T> result(x.rows(), x.cols());
                    for (size_t r = 0; r < x.rows(); ++r) {
                        for (size_t c = 0; c < x.cols(); ++c) {
                            result(r, c) = T(1) / (T(1) + std::exp(-x(r, c)));
                        }
                    }
                    return result;
                }

                // Backward Pass: Calculates the derivative for backpropagation
                template <typename T>
                Core::Matrix<T> Sigmoid<T>::backward(const Core::Matrix<T>& output, const Core::Matrix<T>& upstream_gradient) {
                    Core::Matrix<T> dx(output.rows(), output.cols());
                    for (size_t r = 0; r < output.rows(); ++r) {
                        for (size_t c = 0; c < output.cols(); ++c) {
                            T s = output(r, c);
                            dx(r, c) = upstream_gradient(r, c) * s * (T(1) - s);
                        }
                    }
                    return dx;
                }

            }
        }
    }
}