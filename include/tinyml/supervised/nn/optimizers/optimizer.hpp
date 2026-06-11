#pragma once
#include "tinyml/core/matrix.hpp"

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Optimizers {

                template <typename T>
                class Optimizer {
                public:
                    virtual ~Optimizer() = default;
                    
                    // Forces all child optimizers to implement a standardized weight update pass
                    virtual void update(Core::Matrix<T>& weights, Core::Matrix<T>& bias,
                                        const Core::Matrix<T>& dW, const Core::Matrix<T>& db) = 0;
                };

            }
        }
    }
}