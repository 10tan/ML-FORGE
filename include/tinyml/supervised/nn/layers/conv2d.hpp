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
        class Conv2D : public Layer<T> {
        private:
            size_t m_in_channels;
            size_t m_out_channels;
            size_t m_kernel_size;
            size_t m_stride;
            size_t m_padding;

            Core::Tensor<T> m_filters; // Shape: [out_channels, in_channels, kernel_size, kernel_size]
            Core::Tensor<T> m_biases;  // Shape: [out_channels]

            // Parameter gradient accumulators
            Core::Tensor<T> m_dFilters;
            Core::Tensor<T> m_dBiases;

            // Execution state caches for backpropagation paths
            Core::Tensor<T> m_input_cache;
            std::vector<size_t> m_input_shape;

        public:
            Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
                : m_in_channels(in_channels), m_out_channels(out_channels), 
                  m_kernel_size(kernel_size), m_stride(stride), m_padding(padding) {
                initialize_parameters();
            }

            /**
             * @brief Initializes filter weights using He/Kaiming Normal distribution and sets biases to zero.
             */
            void initialize_parameters() {
                std::vector<size_t> f_shape = {m_out_channels, m_in_channels, m_kernel_size, m_kernel_size};
                std::vector<size_t> b_shape = {m_out_channels, 1, 1, 1}; // Shaped for easy broad element adding

                size_t total_filter_elements = m_out_channels * m_in_channels * m_kernel_size * m_kernel_size;
                std::vector<T> f_data(total_filter_elements);
                std::vector<T> b_data(m_out_channels, T(0));

                // Kaiming Normal limit: sqrt(2.0 / fan_in) where fan_in = in_channels * kernel_size * kernel_size
                T fan_in = static_cast<T>(m_in_channels * m_kernel_size * m_kernel_size);
                T stddev = std::sqrt(T(2.0) / fan_in);

                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

                for (size_t i = 0; i < f_data.size(); ++i) {
                    f_data[i] = static_cast<T>(dist(gen));
                }

                m_filters = Core::Tensor<T>(f_shape, f_data);
                m_biases = Core::Tensor<T>(b_shape, b_data);

                m_dFilters = Core::Tensor<T>(f_shape, std::vector<T>(f_data.size(), T(0)));
                m_dBiases = Core::Tensor<T>(b_shape, std::vector<T>(b_data.size(), T(0)));
            }

            /**
             * @brief Implements sliding-window cross-correlation with stride configurations.
             */
            Core::Tensor<T> forward(const Core::Tensor<T>& input) override {
                m_input_cache = input;
                m_input_shape = input.shape();

                if (m_input_shape.size() != 4 || m_input_shape[1] != m_in_channels) {
                    throw std::runtime_error("Conv2D Forward Error: Input channel structural mismatch.");
                }

                size_t batch = m_input_shape[0];
                size_t in_h = m_input_shape[2];
                size_t in_w = m_input_shape[3];

                // Determine output coordinate limits
                size_t out_h = (in_h - m_kernel_size + 2 * m_padding) / m_stride + 1;
                size_t out_w = (in_w - m_kernel_size + 2 * m_padding) / m_stride + 1;

                std::vector<size_t> out_shape = {batch, m_out_channels, out_h, out_w};
                std::vector<T> out_data(batch * m_out_channels * out_h * out_w, T(0));
                Core::Tensor<T> output(out_shape, out_data);

                // Helper lambda to safely extract padded input values inline
                auto get_padded_val = [&](size_t b, size_t c, int r, int c_idx) -> T {
                    int raw_r = r - static_cast<int>(m_padding);
                    int raw_c = c_idx - static_cast<int>(m_padding);
                    if (raw_r < 0 || raw_r >= static_cast<int>(in_h) || raw_c < 0 || raw_c >= static_cast<int>(in_w)) {
                        return T(0); // Zero-padding logic boundary
                    }
                    return input({b, c, static_cast<size_t>(raw_r), static_cast<size_t>(raw_c)});
                };

                // Core 2D Convolution standard loop execution sequence
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t o = 0; o < m_out_channels; ++o) {
                        for (size_t i = 0; i < out_h; ++i) {
                            for (size_t j = 0; j < out_w; ++j) {
                                
                                T scalar_accumulation = T(0);
                                size_t start_r = i * m_stride;
                                size_t start_c = j * m_stride;

                                for (size_t c = 0; c < m_in_channels; ++c) {
                                    for (size_t kh = 0; kh < m_kernel_size; ++kh) {
                                        for (size_t kw = 0; kw < m_kernel_size; ++kw) {
                                            T val = get_padded_val(b, c, start_r + kh, start_c + kw);
                                            scalar_accumulation += val * m_filters({o, c, kh, kw});
                                        }
                                    }
                                }
                                output({b, o, i, j}) = scalar_accumulation + m_biases({o, 0, 0, 0});
                            }
                        }
                    }
                }

                return output;
            }

            /**
             * @brief Computes analytical spatial gradients via transposed convolutional routing.
             */
            Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) override {
                size_t batch = m_input_shape[0];
                size_t in_h = m_input_shape[2];
                size_t in_w = m_input_shape[3];

                auto grad_shape = output_gradient.shape();
                size_t grad_h = grad_shape[2];
                size_t grad_w = grad_shape[3];

                // Instantiate fresh, zeroed tensor spaces for input and weight derivatives
                std::vector<T> in_grad_data(batch * m_in_channels * in_h * in_w, T(0));
                Core::Tensor<T> input_gradient(m_input_shape, in_grad_data);

                std::vector<T> dF_zero(m_filters.shape()[0] * m_filters.shape()[1] * m_kernel_size * m_kernel_size, T(0));
                std::vector<T> dB_zero(m_out_channels, T(0));
                m_dFilters = Core::Tensor<T>(m_filters.shape(), dF_zero);
                m_dBiases = Core::Tensor<T>(m_biases.shape(), dB_zero);

                auto get_padded_in_grad_ref = [&](size_t b, size_t c, int r, int c_idx) -> T* {
                    int raw_r = r - static_cast<int>(m_padding);
                    int raw_c = c_idx - static_cast<int>(m_padding);
                    if (raw_r < 0 || raw_r >= static_cast<int>(in_h) || raw_c < 0 || raw_c >= static_cast<int>(in_w)) {
                        return nullptr; // Gradients landing on padded coordinates are discarded
                    }
                    return &input_gradient({b, c, static_cast<size_t>(raw_r), static_cast<size_t>(raw_c)});
                };

                auto get_padded_input_val = [&](size_t b, size_t c, int r, int c_idx) -> T {
                    int raw_r = r - static_cast<int>(m_padding);
                    int raw_c = c_idx - static_cast<int>(m_padding);
                    if (raw_r < 0 || raw_r >= static_cast<int>(in_h) || raw_c < 0 || raw_c >= static_cast<int>(in_w)) {
                        return T(0);
                    }
                    return m_input_cache({b, c, static_cast<size_t>(raw_r), static_cast<size_t>(raw_c)});
                };

                // Map derivatives across spatial coordinate alignments
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t o = 0; o < m_out_channels; ++o) {
                        for (size_t i = 0; i < grad_h; ++i) {
                            for (size_t j = 0; j < grad_w; ++j) {
                                
                                T dL_dY = output_gradient({b, o, i, j});
                                m_dBiases({o, 0, 0, 0}) += dL_dY;

                                size_t start_r = i * m_stride;
                                size_t start_c = j * m_stride;

                                for (size_t c = 0; c < m_in_channels; ++c) {
                                    for (size_t kh = 0; kh < m_kernel_size; ++kh) {
                                        for (size_t kw = 0; kw < m_kernel_size; ++kw) {
                                            
                                            // Update filter/weight gradients: dW = X * dY
                                            m_dFilters({o, c, kh, kw}) += get_padded_input_val(b, c, start_r + kh, start_c + kw) * dL_dY;

                                            // Backpropagate input gradients: dX = dY * W
                                            T* in_grad_ptr = get_padded_in_grad_ref(b, c, start_r + kh, start_c + kw);
                                            if (in_grad_ptr != nullptr) {
                                                *in_grad_ptr += dL_dY * m_filters({o, c, kh, kw});
                                            }
                                        }
                                    }
                                }

                            }
                        }
                    }
                }

                return input_gradient;
            }

            /**
             * @brief Updates layer parameters via element-wise gradient steps.
             */
            void update_parameters(T learning_rate) override {
                // 🎯 FIX: Loop through multi-dimensional filter coordinates to update weights mutably
                for (size_t o = 0; o < m_out_channels; ++o) {
                    for (size_t c = 0; c < m_in_channels; ++c) {
                        for (size_t kh = 0; kh < m_kernel_size; ++kh) {
                            for (size_t kw = 0; kw < m_kernel_size; ++kw) {
                                m_filters({o, c, kh, kw}) -= learning_rate * m_dFilters({o, c, kh, kw});
                            }
                        }
                    }
                }
                for (size_t o = 0; o < m_out_channels; ++o) {
                    m_biases({o, 0, 0, 0}) -= learning_rate * m_dBiases({o, 0, 0, 0});
                }
            }

            std::string layer_type() const override { return "Conv2D (Spatial Correlation)"; }

            // Accessors for architectural debugging and optimization tests
            const Core::Tensor<T>& filters() const { return m_filters; }
            const Core::Tensor<T>& biases() const { return m_biases; }
        };

    }
}