// src/supervised/nn/activations/relu.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                Core::Matrix<T> ReLU<T>::forward(const Core::Matrix<T>& input) {
                    Core::Matrix<T> result(input.rows(), input.cols());
                    for (size_t r = 0; r < input.rows(); ++r) {
                        for (size_t c = 0; c < input.cols(); ++c) {
                            result(r, c) = input(r, c) > T(0) ? input(r, c) : T(0);
                        }
                    }
                    return result;
                }

                template <typename T>
                Core::Matrix<T> ReLU<T>::backward(const Core::Matrix<T>& input, const Core::Matrix<T>& upstream_gradient) {
                    Core::Matrix<T> downstream_gradient(input.rows(), input.cols());
                    for (size_t r = 0; r < input.rows(); ++r) {
                        for (size_t c = 0; c < input.cols(); ++c) {
                            downstream_gradient(r, c) = input(r, c) > T(0) ? upstream_gradient(r, c) : T(0);
                        }
                    }
                    return downstream_gradient;
                }

            }
        }
    }
}