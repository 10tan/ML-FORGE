#pragma once
#include "tinyml/core/matrix.hpp"
#include <cmath>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class Tanh {
                public:
                    Tanh() = default;

                    Core::Matrix<T> forward(const Core::Matrix<T>& src) const {
                        Core::Matrix<T> dst(src.rows(), src.cols());
                        for (size_t i = 0; i < src.rows(); ++i) {
                            for (size_t j = 0; j < src.cols(); ++j) {
                                dst(i, j) = std::tanh(src(i, j));
                            }
                        }
                        return dst;
                    }

                    Core::Matrix<T> backward(const Core::Matrix<T>& out_grad, const Core::Matrix<T>& output) const {
                        Core::Matrix<T> in_grad(out_grad.rows(), out_grad.cols());
                        for (size_t i = 0; i < out_grad.rows(); ++i) {
                            for (size_t j = 0; j < out_grad.cols(); ++j) {
                                T val = output(i, j);
                                // Derivative of tanh(x) is 1 - tanh^2(x)
                                in_grad(i, j) = out_grad(i, j) * (T(1.0) - val * val);
                            }
                        }
                        return in_grad;
                    }
                };

            }
        }
    }
}