#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class TSNE {
        private:
            size_t m_components;
            T m_perplexity;
            size_t m_max_iters;
            T m_lr;

            // Helper: Compute flat squared Euclidean distance between two matrix rows
            T squared_dist(const Core::Matrix<T>& mat, size_t r1, size_t r2) const {
                T sum = 0;
                for (size_t j = 0; j < mat.cols(); ++j) {
                    T diff = mat(r1, j) - mat(r2, j);
                    sum += diff * diff;
                }
                return sum;
            }

        public:
            TSNE(size_t n_components = 2, T perplexity = T(30.0), size_t max_iters = 200, T learning_rate = T(10.0))
                : m_components(n_components), m_perplexity(perplexity), m_max_iters(max_iters), m_lr(learning_rate) {}

            /**
             * @brief Fits and transforms high-dimensional data points down to an embedded low-dimensional space
             */
            Core::Matrix<T> fit_transform(const Core::Matrix<T>& X, unsigned int seed = 42) {
                size_t n = X.rows();
                
                // 1. Compute High-Dimensional Probabilities (P-Matrix)
                // For a lightweight scratch engine, we approximate Gaussian affinities with a fixed scale variance (sigma^2 = 1.f)
                Core::Matrix<T> P(n, n);
                for (size_t i = 0; i < n; ++i) {
                    T row_sum = 0;
                    for (size_t j = 0; j < n; ++j) {
                        if (i == j) continue;
                        // Conditional probability: exp(-||x_i - x_j||^2 / 2)
                        P(i, j) = std::exp(-squared_dist(X, i, j) / T(2.0));
                        row_sum += P(i, j);
                    }
                    if (row_sum < T(1e-9)) row_sum = T(1);
                    for (size_t j = 0; j < n; ++j) {
                        if (i != j) P(i, j) /= row_sum;
                    }
                }

                // Symmetrize probabilities: p_ij = (p_j|i + p_i|j) / 2n
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = i + 1; j < n; ++j) {
                        T sym_val = (P(i, j) + P(j, i)) / (T(2.0) * static_cast<T>(n));
                        if (sym_val < T(1e-12)) sym_val = T(1e-12); // Stability floor
                        P(i, j) = sym_val;
                        P(j, i) = sym_val;
                    }
                }

                // 2. Initialize Low-Dimensional Embedding Space randomly (with small values)
                Core::Matrix<T> Y(n, m_components);
                std::mt19937 rng(seed);
                std::normal_distribution<T> dist(0.0, 1e-4);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < m_components; ++j) {
                        Y(i, j) = dist(rng);
                    }
                }

                // Gradient step matrix container
                Core::Matrix<T> dY(n, m_components);

                // 3. Optimization loops minimizing KL Divergence via Gradient Descent
                for (size_t iter = 0; iter < m_max_iters; ++iter) {
                    // Compute Low-Dimensional Student-t Joint Probabilities (Q-Matrix)
                    Core::Matrix<T> Q(n, n);
                    T sum_q = 0;
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            if (i == j) continue;
                            // Student-t mapping: (1 + ||y_i - y_j||^2)^-1
                            Q(i, j) = T(1.0) / (T(1.0) + squared_dist(Y, i, j));
                            sum_q += Q(i, j);
                        }
                    }
                    if (sum_q < T(1e-9)) sum_q = T(1);

                    // Normalize Q matrix
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            if (i != j) Q(i, j) /= sum_q;
                        }
                    }

                    // Compute Gradients: dC/dy_i = 4 * sum_j ((P_ij - Q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^-1)
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t d = 0; d < m_components; ++d) {
                            T grad_sum = 0;
                            for (size_t j = 0; j < n; ++j) {
                                if (i == j) continue;
                                T mult = (P(i, j) - Q(i, j)) * (T(1.0) / (T(1.0) + squared_dist(Y, i, j)));
                                grad_sum += mult * (Y(i, d) - Y(j, d));
                            }
                            dY(i, d) = T(4.0) * grad_sum;
                        }
                    }

                    // Apply Gradient Descent Updates directly to Embedding
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t d = 0; d < m_components; ++d) {
                            Y(i, d) -= m_lr * dY(i, d);
                        }
                    }
                }

                return Y;
            }
        };

    }
}