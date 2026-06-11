#pragma once
#include "tinyml/supervised/nn/layers/layer.hpp"
#include <stdexcept>
#include <string>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class Flatten : public Layer<T> {
        private:
            // Cache the original 4D shape during forward pass to restore it during backprop
            std::vector<size_t> m_input_shape_cache;

        public:
            Flatten() = default;

            /**
             * @brief Flattens a 4D feature map space into a unified 2D batch matrix layout.
             */
            Core::Tensor<T> forward(const Core::Tensor<T>& input) override {
                m_input_shape_cache = input.shape();
                
                if (m_input_shape_cache.size() < 2) {
                    throw std::runtime_error("Flatten Layer Forward Error: Input tensor must have at least 2 dimensions.");
                }

                size_t batch_size = m_input_shape_cache[0];
                
                // Calculate the total remaining features across channels, rows, and columns
                size_t flattened_dim = 1;
                for (size_t i = 1; i < m_input_shape_cache.size(); ++i) {
                    flattened_dim *= m_input_shape_cache[i];
                }

                // Construct the 2D output structure reusing the underlying flat vector mapping
                std::vector<size_t> output_shape = {batch_size, flattened_dim};
                return Core::Tensor<T>(output_shape, input.data());
            }

            /**
             * @brief Restores the incoming 2D matrix gradient back to its original 4D shape.
             */
            Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) override {
                auto grad_shape = output_gradient.shape();
                
                if (grad_shape.size() != 2) {
                    throw std::runtime_error("Flatten Layer Backward Error: Downstream gradient must be 2D.");
                }

                // Invert the flattening operation by sending the vector back using cached input shapes
                return Core::Tensor<T>(m_input_shape_cache, output_gradient.data());
            }

            std::string layer_type() const override { return "Flatten (4D -> 2D)"; }
        };

    }
}