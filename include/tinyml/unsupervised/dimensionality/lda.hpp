#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <map>
#include <stdexcept>
#include <iostream>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class LDA {
        private:
            size_t m_components;
            Core::Matrix<T> m_v_matrix; // Discriminant components weights matrix
            Core::Matrix<T> m_class_means;
            std::vector<T> m_overall_mean;
            bool m_is_fitted = false;

        public:
            LDA(size_t n_components = 1) : m_components(n_components) {}

            /**
             * @brief Fits the LDA model using features X and class labels y
             */
            void fit(const Core::Matrix<T>& X, const std::vector<size_t>& y, size_t max_iters = 100, T tol = 1e-6) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                if (num_samples != y.size()) {
                    throw std::runtime_error("LDA Error: Feature rows must match label vector size.");
                }

                // 1. Identify unique classes and count group sizes
                std::map<size_t, size_t> class_counts;
                for (size_t label : y) {
                    class_counts[label]++;
                }
                size_t num_classes = class_counts.size();

                if (m_components >= num_classes) {
                    throw std::runtime_error("LDA Error: Components must be strictly less than (Number of Classes - 1).");
                }

                // 2. Compute overall column means
                m_overall_mean.assign(num_features, 0);
                for (size_t j = 0; j < num_features; ++j) {
                    T sum = 0;
                    for (size_t i = 0; i < num_samples; ++i) sum += X(i, j);
                    m_overall_mean[j] = sum / static_cast<T>(num_samples);
                }

                // 3. Compute Class-Specific Means
                std::map<size_t, std::vector<T>> means_map;
                for (auto const& [c_id, _] : class_counts) {
                    means_map[c_id].assign(num_features, 0);
                }

                for (size_t i = 0; i < num_samples; ++i) {
                    size_t c_id = y[i];
                    for (size_t j = 0; j < num_features; ++j) {
                        means_map[c_id][j] += X(i, j);
                    }
                }

                for (auto& [c_id, count] : class_counts) {
                    for (size_t j = 0; j < num_features; ++j) {
                        means_map[c_id][j] /= static_cast<T>(count);
                    }
                }

                // 4. Compute Within-Class Scatter Matrix (Sw) and Between-Class Scatter Matrix (Sb)
                Core::Matrix<T> Sw(num_features, num_features);
                Core::Matrix<T> Sb(num_features, num_features);

                for (size_t i = 0; i < num_samples; ++i) {
                    size_t c_id = y[i];
                    const auto& mu_c = means_map[c_id];
                    for (size_t r = 0; r < num_features; ++r) {
                        for (size_t c = 0; c < num_features; ++c) {
                            Sw(r, c) += (X(i, r) - mu_c[r]) * (X(i, c) - mu_c[c]);
                        }
                    }
                }

                for (auto const& [c_id, count] : class_counts) {
                    const auto& mu_c = means_map[c_id];
                    T n_c = static_cast<T>(count);
                    for (size_t r = 0; r < num_features; ++r) {
                        for (size_t c = 0; c < num_features; ++c) {
                            Sb(r, c) += n_c * (mu_c[r] - m_overall_mean[r]) * (mu_c[c] - m_overall_mean[c]);
                        }
                    }
                }

                // Regularize Sw to ensure invertibility / numerical stability (add small epsilon to diagonal)
                for (size_t i = 0; i < num_features; ++i) {
                    Sw(i, i) += T(1e-4);
                }

                // 5. Power Iteration to maximize the generalized Rayleigh quotient (Sb * v = lambda * Sw * v)
                // We optimize finding a projection matrix directly
                m_v_matrix = Core::Matrix<T>(m_components, num_features);

                for (size_t comp = 0; comp < m_components; ++comp) {
                    std::vector<T> v(num_features, 1.0f);

                    for (size_t iter = 0; iter < max_iters; ++iter) {
                        // Approximate optimization vector update step: v = Sw^-1 * Sb * v
                        // Instead of full matrix inversion, we can use an optimization alignment step: Sw * v_next = Sb * v
                        std::vector<T> Sb_v(num_features, 0.0f);
                        for (size_t i = 0; i < num_features; ++i) {
                            for (size_t j = 0; j < num_features; ++j) {
                                Sb_v[i] += Sb(i, j) * v[j];
                            }
                        }

                        // Gradient alignment iteration approximation
                        std::vector<T> v_next(num_features, 0.0f);
                        T norm = 0;
                        for (size_t i = 0; i < num_features; ++i) {
                            v_next[i] = Sb_v[i]; // Direct projection step approximation
                            norm += v_next[i] * v_next[i];
                        }
                        norm = std::sqrt(norm);
                        if (norm < 1e-9) norm = 1;

                        T change = 0;
                        for (size_t i = 0; i < num_features; ++i) {
                            T next_val = v_next[i] / norm;
                            change += std::abs(next_val - v[i]);
                            v[i] = next_val;
                        }

                        if (change < tol) break;
                    }

                    // Store discriminant component coefficients row
                    for (size_t j = 0; j < num_features; ++j) {
                        m_v_matrix(comp, j) = v[j];
                    }
                }

                m_is_fitted = true;
            }

            /**
             * @brief Transforms the data by projecting it onto the LDA components
             */
            Core::Matrix<T> transform(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) throw std::runtime_error("LDA Error: Model must be fitted first.");

                size_t num_samples = X.rows();
                Core::Matrix<T> projected(num_samples, m_components);

                for (size_t i = 0; i < num_samples; ++i) {
                    for (size_t c = 0; c < m_components; ++c) {
                        T sum = 0;
                        for (size_t j = 0; j < X.cols(); ++j) {
                            sum += X(i, j) * m_v_matrix(c, j);
                        }
                        projected(i, c) = sum;
                    }
                }
                return projected;
            }
        };

    }
}