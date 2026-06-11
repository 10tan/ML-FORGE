#pragma once
#include "tinyml/supervised/nn/layers/layer.hpp"
#include <stdexcept>
#include <vector>
#include <string>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class MaxPooling2D : public Layer<T> {
        private:
            size_t m_pool_size;
            size_t m_stride;

            // Cache input tensor and coordinate lookup tables to perform max-gradient routing during backprop
            Core::Tensor<T> m_input_cache;
            std::vector<size_t> m_input_shape;

        public:
            MaxPooling2D(size_t pool_size = 2, size_t stride = 2)
                : m_pool_size(pool_size), m_stride(stride) {}

            Core::Tensor<T> forward(const Core::Tensor<T>& input) override {
                m_input_cache = input;
                m_input_shape = input.shape();

                if (m_input_shape.size() != 4) {
                    throw std::runtime_error("MaxPooling2D Error: Input tensor must be 4D [B, C, H, W].");
                }

                size_t batch = m_input_shape[0];
                size_t channels = m_input_shape[1];
                size_t in_h = m_input_shape[2];
                size_t in_w = m_input_shape[3];

                // Calculate output spatial limits
                size_t out_h = (in_h - m_pool_size) / m_stride + 1;
                size_t out_w = (in_w - m_pool_size) / m_stride + 1;

                std::vector<size_t> out_shape = {batch, channels, out_h, out_w};
                std::vector<T> out_data(batch * channels * out_h * out_w, T(0));

                for (size_t b = 0; b < batch; ++b) {
                    for (size_t c = 0; c < channels; ++c) {
                        for (size_t i = 0; i < out_h; ++i) {
                            for (size_t j = 0; j < out_w; ++j) {
                                
                                size_t start_h = i * m_stride;
                                size_t start_w = j * m_stride;
                                T max_val = input({b, c, start_h, start_w});

                                // Slide across the localized window matrix pool
                                for (size_t ph = 0; ph < m_pool_size; ++ph) {
                                    for (size_t pw = 0; pw < m_pool_size; ++pw) {
                                        T curr_val = input({b, c, start_h + ph, start_w + pw});
                                        if (curr_val > max_val) {
                                            max_val = curr_val;
                                        }
                                    }
                                }
                                
                                // Flattened mapping index store: out_data[b][c][i][j]
                                size_t out_idx = b * (channels * out_h * out_w) + c * (out_h * out_w) + i * out_w + j;
                                out_data[out_idx] = max_val;
                            }
                        }
                    }
                }

                return Core::Tensor<T>(out_shape, out_data);
            }

            Core::Tensor<T> backward(const Core::Tensor<T>& output_gradient) override {
                size_t batch = m_input_shape[0];
                size_t channels = m_input_shape[1];
                size_t in_h = m_input_shape[2];
                size_t in_w = m_input_shape[3];

                auto grad_shape = output_gradient.shape();
                size_t out_h = grad_shape[2];
                size_t out_w = grad_shape[3];

                // Instantiate an empty input gradient array matching the raw 4D input boundaries
                std::vector<T> in_grad_data(batch * channels * in_h * in_w, T(0));
                Core::Tensor<T> input_gradient(m_input_shape, in_grad_data);

                for (size_t b = 0; b < batch; ++b) {
                    for (size_t c = 0; c < channels; ++c) {
                        for (size_t i = 0; i < out_h; ++i) {
                            for (size_t j = 0; j < out_w; ++j) {
                                
                                size_t start_h = i * m_stride;
                                size_t start_w = j * m_stride;
                                
                                size_t max_h = start_h;
                                size_t max_w = start_w;
                                T max_val = m_input_cache({b, c, start_h, start_w});

                                // Relocate the precise maximum element index track coordinate
                                for (size_t ph = 0; ph < m_pool_size; ++ph) {
                                    for (size_t pw = 0; pw < m_pool_size; ++pw) {
                                        size_t cur_h = start_h + ph;
                                        size_t cur_w = start_w + pw;
                                        T curr_val = m_input_cache({b, c, cur_h, cur_w});
                                        if (curr_val > max_val) {
                                            max_val = curr_val;
                                            max_h = cur_h;
                                            max_w = cur_w;
                                        }
                                    }
                                }

                                // Fetch the downstream loss gradient error trace element
                                T downstream_grad = output_gradient({b, c, i, j});
                                
                                // Route the error back ONLY to the winner cell coordinate track
                                input_gradient({b, c, max_h, max_w}) += downstream_grad;
                            }
                        }
                    }
                }

                return input_gradient;
            }

            std::string layer_type() const override { return "MaxPooling2D"; }
        };

    }
}