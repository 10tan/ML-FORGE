#pragma once
#include "tinyml/core/tensor.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace MLForge {
    namespace NeuralNetwork {

        template <typename T>
        class SoftmaxCrossEntropy {
        private:
            Core::Tensor<T> m_probabilities;

        public:
            SoftmaxCrossEntropy() = default;

            /**
             * @brief Computes forward loss and caches probabilities.
             * @param logits Shape: [Batch, 10] raw scores from the final Dense layer.
             * @param targets Vector of true class indices (size == Batch).
             */
            T forward(const Core::Tensor<T>& logits, const std::vector<size_t>& targets) {
                auto shape = logits.shape();
                size_t batch_size = shape[0];
                size_t num_classes = shape[1];

                std::vector<T> prob_data(batch_size * num_classes, T(0));
                T total_loss = T(0);

                for (size_t b = 0; b < batch_size; ++b) {
                    // 1. Find max logit for numerical stability (prevents overflow in exp)
                    T max_logit = logits({b, 0});
                    for (size_t c = 1; c < num_classes; ++c) {
                        max_logit = std::max(max_logit, logits({b, c}));
                    }

                    // 2. Compute exponentials and sum
                    T sum_exp = T(0);
                    std::vector<T> exps(num_classes);
                    for (size_t c = 0; c < num_classes; ++c) {
                        exps[c] = std::exp(logits({b, c}) - max_logit);
                        sum_exp += exps[c];
                    }

                    // 3. Compute stable probabilities and map loss
                    for (size_t c = 0; c < num_classes; ++c) {
                        size_t idx = b * num_classes + c;
                        prob_data[idx] = exps[c] / sum_exp;
                    }

                    size_t target_class = targets[b];
                    T target_prob = prob_data[b * num_classes + target_class];
                    // Clip probability to avoid log(0)
                    target_prob = std::max(target_prob, static_cast<T>(1e-15));
                    total_loss += -std::log(target_prob);
                }

                m_probabilities = Core::Tensor<T>(shape, prob_data);
                return total_loss / static_cast<T>(batch_size);
            }

            /**
             * @brief Computes downstream gradient: dL/dLogits = (probabilities - targets) / Batch
             */
            Core::Tensor<T> backward(const std::vector<size_t>& targets) {
                auto shape = m_probabilities.shape();
                size_t batch_size = shape[0];
                size_t num_classes = shape[1];

                std::vector<T> grad_data = m_probabilities.data(); // Starts as raw probabilities

                for (size_t b = 0; b < batch_size; ++b) {
                    size_t target_class = targets[b];
                    // Subtract 1 from the true active target index position (One-Hot derivative)
                    grad_data[b * num_classes + target_class] -= T(1);

                    // Normalize gradient across the batch scale footprint
                    for (size_t c = 0; c < num_classes; ++c) {
                        grad_data[b * num_classes + c] /= static_cast<T>(batch_size);
                    }
                }

                return Core::Tensor<T>(shape, grad_data);
            }

            const Core::Tensor<T>& probabilities() const { return m_probabilities; }
        };

    }
}