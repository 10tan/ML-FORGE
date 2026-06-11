#pragma once
#include "tinyml/core/matrix.hpp"
#include <algorithm>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class LeakyReLU {
                private:
                    T m_alpha; // Negative slope coefficient

                public:
                    LeakyReLU(T alpha = T(0.01)) : m_alpha(alpha) {}

                    Core::Matrix<T> forward(const Core::Matrix<T>& src) const {
                        Core::Matrix<T> dst(src.rows(), src.cols());
                        for (size_t i = 0; i < src.rows(); ++i) {
                            for (size_t j = 0; j < src.cols(); ++j) {
                                T val = src(i, j);
                                dst(i, j) = (val > T(0.0)) ? val : m_alpha * val;
                            }
                        }
                        return dst;
                    }

                    Core::Matrix<T> backward(const Core::Matrix<T>& out_grad, const Core::Matrix<T>& output) const {
                        Core::Matrix<T> in_grad(out_grad.rows(), out_grad.cols());
                        for (size_t i = 0; i < out_grad.rows(); ++i) {
                            for (size_t j = 0; j < out_grad.cols(); ++j) {
                                // If the forward output was positive, derivative is 1.0; else it's alpha.
                                T slope = (output(i, j) > T(0.0)) ? T(1.0) : m_alpha;
                                in_grad(i, j) = out_grad(i, j) * slope;
                            }
                        }
                        return in_grad;
                    }
                };

            }
        }
    }
}