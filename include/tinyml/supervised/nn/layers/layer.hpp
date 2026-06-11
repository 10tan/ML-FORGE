#pragma once
#include "tinyml/core/tensor.hpp"
#include <memory>
#include <string>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class Layer {
        public:
            virtual ~Layer() = default;

            /**
             * @brief Executes the forward execution loop of the network layer.
             * @param input The incoming tensor profile (cached internally if required for backprop steps).
             * @return Core::Tensor<T> The transformed feature output tensor.
             */
            virtual Core::Tensor<T> forward(const Core::Tensor<T>& input) = 0;

            /**
             * @brief Runs backward error propagation to evaluate gradient distributions.
             * @param output_gradient The gradient of the loss function with respect to this layer's output.
             * @return Core::Tensor<T> The gradient of the loss function with respect to this layer's input.
             */
            virtual Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) = 0;

            /**
             * @brief Adjusts parameters using calculated gradient matrices.
             * @param learning_rate Optimization step configuration scalar.
             */
            virtual void update_parameters(T learning_rate) {}

            /**
             * @brief Structural string identifier tracking layer properties.
             */
            virtual std::string layer_type() const = 0;

            // Add this inside your virtual Layer class definition
            virtual void serialize(std::ofstream& os) const = 0;
            virtual void deserialize(std::ifstream& is) = 0;
        };

    }
}