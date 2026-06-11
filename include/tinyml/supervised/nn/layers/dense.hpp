#pragma once
#include "tinyml/supervised/nn/layers/layer.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class Dense : public Layer<T> {
        private:
            size_t m_input_dim;
            size_t m_output_dim;

            Core::Tensor<T> m_weights; // Structural shape bounds: [input_dim, output_dim]
            Core::Tensor<T> m_biases;  // Structural shape bounds: [1, output_dim]

            // Gradient parameter stores
            Core::Tensor<T> m_dW;
            Core::Tensor<T> m_db;

            // Input activation snapshot cache tracked for backward parameter sweeps
            Core::Tensor<T> m_input_cache;

        public:
            Dense(size_t input_dim, size_t output_dim)
                : m_input_dim(input_dim), m_output_dim(output_dim) {
                initialize_parameters();
            }

            /**
             * @brief Sets up weights using Xavier Normal allocation rules and clears biases.
             */
            void initialize_parameters() {
                std::vector<size_t> w_shape = {m_input_dim, m_output_dim};
                std::vector<size_t> b_shape = {1, m_output_dim};

                std::vector<T> w_data(m_input_dim * m_output_dim);
                std::vector<T> b_data(m_output_dim, T(0));

                // Standard deviation target: sqrt(2.0 / (input_dim + output_dim))
                T stddev = std::sqrt(T(2.0) / static_cast<T>(m_input_dim + m_output_dim));
                
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

                for (size_t i = 0; i < w_data.size(); ++i) {
                    w_data[i] = static_cast<T>(dist(gen));
                }

                m_weights = Core::Tensor<T>(w_shape, w_data);
                m_biases = Core::Tensor<T>(b_shape, b_data);
                
                m_dW = Core::Tensor<T>(w_shape, std::vector<T>(w_data.size(), T(0)));
                m_db = Core::Tensor<T>(b_shape, std::vector<T>(b_data.size(), T(0)));
            }

            /**
             * @brief Evaluates Z = XW + b inline.
             */
            Core::Tensor<T> forward(const Core::Tensor<T>& input) override {
                auto shape = input.shape();
                if (shape.size() != 2 || shape[1] != m_input_dim) {
                    throw std::runtime_error("Dense Layer Forward Error: Input size mismatch coordinates.");
                }

                m_input_cache = input;
                size_t batch_size = shape[0];

                std::vector<size_t> out_shape = {batch_size, m_output_dim};
                std::vector<T> out_data(batch_size * m_output_dim, T(0));

                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t o = 0; o < m_output_dim; ++o) {
                        T linear_combination = T(0);
                        for (size_t i = 0; i < m_input_dim; ++i) {
                            linear_combination += input({b, i}) * m_weights({i, o});
                        }
                        out_data[b * m_output_dim + o] = linear_combination + m_biases({0, o});
                    }
                }

                return Core::Tensor<T>(out_shape, out_data);
            }

            /**
             * @brief Computes parameter derivatives and returns input space errors.
             */
            Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) override {
                auto grad_shape = output_gradient.shape();
                size_t batch_size = grad_shape[0];

                std::vector<size_t> in_grad_shape = {batch_size, m_input_dim};
                std::vector<T> in_grad_data(batch_size * m_input_dim, T(0));
                Core::Tensor<T> input_gradient(in_grad_shape, in_grad_data);

                // Zero out derivative matrices before backprop calculation
                std::vector<T> dW_zero(m_input_dim * m_output_dim, T(0));
                std::vector<T> db_zero(m_output_dim, T(0));
                m_dW = Core::Tensor<T>({m_input_dim, m_output_dim}, dW_zero);
                m_db = Core::Tensor<T>({1, m_output_dim}, db_zero);

                // Run backpropagation loops
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t o = 0; o < m_output_dim; ++o) {
                        T dL_dZ = output_gradient({b, o});
                        
                        m_db({0, o}) += dL_dZ;

                        for (size_t i = 0; i < m_input_dim; ++i) {
                            m_dW({i, o}) += m_input_cache({b, i}) * dL_dZ;
                            input_gradient({b, i}) += dL_dZ * m_weights({i, o});
                        }
                    }
                }

                return input_gradient;
            }

            /**
             * @brief Modifies weight arrays using basic Stochastic Gradient Descent.
             */
            void update_parameters(T learning_rate) override {
                for (size_t i = 0; i < m_input_dim; ++i) {
                    for (size_t o = 0; o < m_output_dim; ++o) {
                        m_weights({i, o}) -= learning_rate * m_dW({i, o});
                    }
                }

                for (size_t o = 0; o < m_output_dim; ++o) {
                    m_biases({0, o}) -= learning_rate * m_db({0, o});
                }
            }

            std::string layer_type() const override { return "Dense (Fully Connected)"; }

            void serialize(std::ofstream& os) const override {
    // 1. Write the shape dimensions so we know how to reconstruct the tensor layout
    auto shape = m_weights.shape();
    size_t shape_size = shape.size();
    os.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    os.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(size_t));

    // 2. Dump the raw binary weight data array
    size_t data_size = m_weights.data().size();
    os.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    os.write(reinterpret_cast<const char*>(m_weights.data().data()), data_size * sizeof(T));
}

void deserialize(std::ifstream& is) override {
    // 1. Read shape dimensions back out
    size_t shape_size = 0;
    is.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    std::vector<size_t> shape(shape_size);
    is.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));

    // 2. Read the raw binary weight data array back in
    size_t data_size = 0;
    is.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    std::vector<T> data(data_size);
    is.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(T));

    // 3. Reconstruct the internal tensor parameter matrix
    m_weights = Core::Tensor<T>(shape, data);
}

            // Weight matrix evaluation lookups
            const Core::Tensor<T>& weights() const { return m_weights; }
            const Core::Tensor<T>& biases() const { return m_biases; }
        };

    }
}