#pragma once
#include "tinyml/core/matrix.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class Softmax {
                public:
                    Softmax() = default;

                    /**
                     * @brief Computes row-wise Softmax probabilities across a 2D sample matrix.
                     */
                    Core::Matrix<T> forward(const Core::Matrix<T>& src) const {
                        size_t rows = src.rows();
                        size_t cols = src.cols();
                        Core::Matrix<T> dst(rows, cols);

                        for (size_t i = 0; i < rows; ++i) {
                            // 1. Find max element in the row for numerical stability
                            T max_val = src(i, 0);
                            for (size_t j = 1; j < cols; ++j) {
                                max_val = std::max(max_val, src(i, j));
                            }

                            // 2. Compute exponentials and sum them up
                            T sum = T(0.0);
                            std::vector<T> exps(cols);
                            for (size_t j = 0; j < cols; ++j) {
                                exps[j] = std::exp(src(i, j) - max_val);
                                sum += exps[j];
                            }

                            // 3. Normalize row probabilities
                            for (size_t j = 0; j < cols; ++j) {
                                dst(i, j) = (sum > T(0.0)) ? (exps[j] / sum) : T(0.0);
                            }
                        }
                        return dst;
                    }

                    /**
                     * @brief Standard derivative placeholder for backpropagation chains.
                     */
                    Core::Matrix<T> backward(const Core::Matrix<T>& out_grad, const Core::Matrix<T>& output) const {
                        // Softmax gradients are typically integrated directly with the Cross-Entropy loss 
                        // derivative (yielding simple pred - true equations), but we provide a direct element pass here.
                        return out_grad; 
                    }
                };

            }
        }
    }
}