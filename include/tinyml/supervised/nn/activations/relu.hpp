#pragma once
#include "tinyml/core/matrix.hpp"

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class ReLU {
                public:
                    // Computes element-wise max(0, x)
                    static Core::Matrix<T> forward(const Core::Matrix<T>& input);

                    // Computes element-wise gradient step based on downstream inputs
                    static Core::Matrix<T> backward(const Core::Matrix<T>& input, const Core::Matrix<T>& upstream_gradient);
                };

            }
        }
    }
}