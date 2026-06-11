#pragma once
#include "tinyml/core/matrix.hpp"
#include <algorithm>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class ReLU {
                public:
                    ReLU() = default;

                    /**
                     * @brief Computes element-wise ReLU forward pass.
                     */
                    Core::Matrix<T> forward(const Core::Matrix<T>& src) const {
                        Core::Matrix<T> dst(src.rows(), src.cols());
                        for (size_t i = 0; i < src.rows(); ++i) {
                            for (size_t j = 0; j < src.cols(); ++j) {
                                T val = src(i, j);
                                dst(i, j) = (val > T(0.0)) ? val : T(0.0);
                            }
                        }
                        return dst;
                    }

                    /**
                     * @brief Computes element-wise scalar ReLU forward pass.
                     */
                    T forward_scalar(T x) const {
                        return (x > T(0.0)) ? x : T(0.0);
                    }

                    /**
                     * @brief Computes backpropagation gradients using the forward layer output.
                     */
                    Core::Matrix<T> backward(const Core::Matrix<T>& out_grad, const Core::Matrix<T>& output) const {
                        Core::Matrix<T> in_grad(out_grad.rows(), out_grad.cols());
                        for (size_t i = 0; i < out_grad.rows(); ++i) {
                            for (size_t j = 0; j < out_grad.cols(); ++j) {
                                // Gradient is out_grad if the activated output was > 0, otherwise 0
                                in_grad(i, j) = (output(i, j) > T(0.0)) ? out_grad(i, j) : T(0.0);
                            }
                        }
                        return in_grad;
                    }
                };

            }
        }
    }
}