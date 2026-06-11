#pragma once
#include "tinyml/core/matrix.hpp"
#include <cmath>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class Sigmoid {
                public:
                    Sigmoid() = default;

                    /**
                     * @brief Computes element-wise Sigmoid forward pass.
                     */
                    Core::Matrix<T> forward(const Core::Matrix<T>& src) const {
                        Core::Matrix<T> dst(src.rows(), src.cols());
                        for (size_t i = 0; i < src.rows(); ++i) {
                            for (size_t j = 0; j < src.cols(); ++j) {
                                dst(i, j) = T(1.0) / (T(1.0) + std::exp(-src(i, j)));
                            }
                        }
                        return dst;
                    }

                    /**
                     * @brief Computes element-wise scalar Sigmoid forward pass.
                     */
                    T forward_scalar(T x) const {
                        return T(1.0) / (T(1.0) + std::exp(-x));
                    }

                    /**
                     * @brief Computes backpropagation gradients using the forward layer output.
                     */
                    Core::Matrix<T> backward(const Core::Matrix<T>& out_grad, const Core::Matrix<T>& output) const {
                        Core::Matrix<T> in_grad(out_grad.rows(), out_grad.cols());
                        for (size_t i = 0; i < out_grad.rows(); ++i) {
                            for (size_t j = 0; j < out_grad.cols(); ++j) {
                                T val = output(i, j);
                                // Derivative: out_grad * sig(x) * (1 - sig(x))
                                in_grad(i, j) = out_grad(i, j) * val * (T(1.0) - val);
                            }
                        }
                        return in_grad;
                    }
                };

            }
        }
    }
}