#pragma once
#include "tinyml/core/matrix.hpp"

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Activations {

                template <typename T>
                class Sigmoid {
                public:
                    // Forward pass: f(x) = 1 / (1 + exp(-x))
                    static Core::Matrix<T> forward(const Core::Matrix<T>& x);
                    
                    // Backward pass: f'(x) = f(x) * (1 - f(x))
                    static Core::Matrix<T> backward(const Core::Matrix<T>& output, const Core::Matrix<T>& upstream_gradient);
                };

            }
        }
    }
}