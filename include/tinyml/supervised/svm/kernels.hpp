#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {
        namespace SVM {

            template <typename T>
            class SVMKernels {
            public:
                enum class KernelType { LINEAR, POLY, RBF };

                /**
                 * @brief Computes the dot product between two raw vector sequences.
                 */
                static T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
                    if (a.size() != b.size()) {
                        throw std::runtime_error("Kernel Error: Vector dimensions must match for dot product.");
                    }
                    T result = T(0.0);
                    for (size_t i = 0; i < a.size(); ++i) {
                        result += a[i] * b[i];
                    }
                    return result;
                }

                /**
                 * @brief Computes the squared Euclidean distance: ||a - b||^2
                 */
                static T squared_distance(const std::vector<T>& a, const std::vector<T>& b) {
                    if (a.size() != b.size()) {
                        throw std::runtime_error("Kernel Error: Vector dimensions must match for distance tracking.");
                    }
                    T dist = T(0.0);
                    for (size_t i = 0; i < a.size(); ++i) {
                        T diff = a[i] - b[i];
                        dist += diff * diff;
                    }
                    return dist;
                }

                /**
                 * @brief Core dispatcher to calculate specialized kernel outputs.
                 */
                static T compute(const std::vector<T>& a, const std::vector<T>& b, 
                                 KernelType type, T gamma = T(0.1), T coef0 = T(0.0), int degree = 3) {
                    switch (type) {
                        case KernelType::LINEAR:
                            return dot_product(a, b);
                        case KernelType::POLY:
                            return std::pow((gamma * dot_product(a, b) + coef0), degree);
                        case KernelType::RBF:
                            return std::exp(-gamma * squared_distance(a, b));
                        default:
                            return T(0.0);
                    }
                }
            };

        }
    }
}