#pragma once
#include "tinyml/supervised/nn/layers/layer.hpp"
#include <algorithm>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class ReLU : public Layer<T> {
        private:
            Core::Tensor<T> m_input_cache;

        public:
            ReLU() = default;

            Core::Tensor<T> forward(const Core::Tensor<T>& input) override {
                m_input_cache = input;
                auto shape = input.shape();
                auto data = input.data(); // Access raw vector
                
                // Forward: f(x) = max(0, x)
                for (size_t i = 0; i < data.size(); ++i) {
                    if (data[i] < T(0)) {
                        data[i] = T(0);
                    }
                }
                return Core::Tensor<T>(shape, data);
            }

            Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) override {
                auto shape = output_gradient.shape();
                auto grad_data = output_gradient.data();
                const auto& input_data = m_input_cache.data();

                // Backward: dL/dx = dL/dy * f'(x), where f'(x) = 1 if x > 0 else 0
                for (size_t i = 0; i < grad_data.size(); ++i) {
                    if (input_data[i] <= T(0)) {
                        grad_data[i] = T(0);
                    }
                }
                return Core::Tensor<T>(shape, grad_data);
            }

            void update_parameters(T) override {
                // ReLU has no learnable parameters
            }

            std::string layer_type() const override { return "ReLU Activation"; }
        };

    }
}