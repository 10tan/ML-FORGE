#pragma once
#include "tinyml/core/matrix.hpp"
#include "kernels.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {
        namespace SVM {

            template <typename T>
            class SVMClassifier {
            private:
                typename SVMKernels<T>::KernelType m_kernel_type;
                T m_C;             // Soft-margin penalty parameter
                T m_gamma;         // RBF scaling factor
                T m_coef0;         // Poly polynomial offset
                int m_degree;      // Poly polynomial degree
                size_t m_max_passes; // Max iteration cycles without alpha updates
                T m_tol;           // KKT numerical tolerance metric

                std::vector<std::vector<T>> m_support_vectors;
                std::vector<T> m_support_vector_labels;
                std::vector<T> m_alphas;
                T m_bias = T(0.0);
                bool m_is_fitted = false;

                T compute_kernel_value(const std::vector<T>& a, const std::vector<T>& b) const {
                    return SVMKernels<T>::compute(a, b, m_kernel_type, m_gamma, m_coef0, m_degree);
                }

                T compute_functional_margin(const std::vector<T>& x) const {
                    T result = m_bias;
                    for (size_t i = 0; i < m_support_vectors.size(); ++i) {
                        result += m_alphas[i] * m_support_vector_labels[i] * compute_kernel_value(m_support_vectors[i], x);
                    }
                    return result;
                }

            public:
                SVMClassifier(typename SVMKernels<T>::KernelType kernel = SVMKernels<T>::KernelType::LINEAR,
                              T C = T(1.0), T gamma = T(0.5), T coef0 = T(0.0), int degree = 3, size_t max_passes = 10, T tol = T(1e-3))
                    : m_kernel_type(kernel), m_C(C), m_gamma(gamma), m_coef0(coef0), m_degree(degree), m_max_passes(max_passes), m_tol(tol) {}

                /**
                 * @brief Solves the dual Lagrangian margin space via Simplified SMO optimization loops.
                 */
                void fit(const Core::Matrix<T>& X, const std::vector<size_t>& y_labels) {
                    size_t n_samples = X.rows();
                    size_t n_features = X.cols();

                    if (n_samples != y_labels.size() || n_samples == 0) {
                        throw std::runtime_error("SVM Error: Row matrix configurations must align with label vector targets.");
                    }

                    // Map structural indices to canonical dual classification targets (-1.0 or +1.0)
                    std::vector<T> y(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        y[i] = (y_labels[i] == 1) ? T(1.0) : T(-1.0);
                    }

                    std::vector<T> alpha_pool(n_samples, T(0.0));
                    m_bias = T(0.0);
                    size_t passes = 0;

                    // Extract matrix data into easy-access vector caches
                    std::vector<std::vector<T>> x_cache(n_samples, std::vector<T>(n_features));
                    for (size_t i = 0; i < n_samples; ++i) {
                        for (size_t j = 0; j < n_features; ++j) {
                            x_cache[i][j] = X(i, j);
                        }
                    }

                    // Main Simplified SMO solver pipeline execution
                    while (passes < m_max_passes) {
                        size_t num_changed_alphas = 0;

                        for (size_t i = 0; i < n_samples; ++i) {
                            T f_i = m_bias;
                            for (size_t k = 0; k < n_samples; ++k) {
                                f_i += alpha_pool[k] * y[k] * compute_kernel_value(x_cache[k], x_cache[i]);
                            }
                            T E_i = f_i - y[i];

                            // KKT Complementary Slackness check boundaries
                            if ((y[i] * E_i < -m_tol && alpha_pool[i] < m_C) || (y[i] * E_i > m_tol && alpha_pool[i] > T(0.0))) {
                                
                                // Random selection loop for independent second dual parameter (j != i)
                                size_t j = i;
                                while (j == i) {
                                    j = rand() % n_samples;
                                }

                                T f_j = m_bias;
                                for (size_t k = 0; k < n_samples; ++k) {
                                    f_j += alpha_pool[k] * y[k] * compute_kernel_value(x_cache[k], x_cache[j]);
                                }
                                T E_j = f_j - y[j];

                                T alpha_i_old = alpha_pool[i];
                                T alpha_j_old = alpha_pool[j];

                                // Determine box constraint bounds
                                T L, H;
                                if (y[i] != y[j]) {
                                    L = std::max(T(0.0), alpha_pool[j] - alpha_pool[i]);
                                    H = std::min(m_C, m_C + alpha_pool[j] - alpha_pool[i]);
                                } else {
                                    L = std::max(T(0.0), alpha_pool[i] + alpha_pool[j] - m_C);
                                    H = std::min(m_C, alpha_pool[i] + alpha_pool[j]);
                                }

                                if (std::abs(L - H) < T(1e-5)) continue;

                                // Compute Kernel Similarity parameters (eta)
                                T k_ii = compute_kernel_value(x_cache[i], x_cache[i]);
                                T k_jj = compute_kernel_value(x_cache[j], x_cache[j]);
                                T k_ij = compute_kernel_value(x_cache[i], x_cache[j]);
                                T eta = T(2.0) * k_ij - k_ii - k_jj;

                                if (eta >= T(0.0)) continue;

                                // Clip updated alpha tracking
                                alpha_pool[j] -= (y[j] * (E_i - E_j)) / eta;
                                alpha_pool[j] = std::max(L, std::min(H, alpha_pool[j]));

                                if (std::abs(alpha_pool[j] - alpha_j_old) < T(1e-5)) continue;

                                alpha_pool[i] += y[i] * y[j] * (alpha_j_old - alpha_pool[j]);

                                // Re-calculate continuous bias term threshold points
                                T b1 = m_bias - E_i - y[i] * (alpha_pool[i] - alpha_i_old) * k_ii - y[j] * (alpha_pool[j] - alpha_j_old) * k_ij;
                                T b2 = m_bias - E_j - y[i] * (alpha_pool[i] - alpha_i_old) * k_ij - y[j] * (alpha_pool[j] - alpha_j_old) * k_jj;

                                if (T(0.0) < alpha_pool[i] && alpha_pool[i] < m_C) m_bias = b1;
                                else if (T(0.0) < alpha_pool[j] && alpha_pool[j] < m_C) m_bias = b2;
                                else m_bias = (b1 + b2) / T(2.0);

                                num_changed_alphas++;
                            }
                        }

                        if (num_changed_alphas == 0) passes++;
                        else passes = 0;
                    }

                    // Cache only valid non-zero support vectors to maximize execution efficiency
                    m_support_vectors.clear();
                    m_support_vector_labels.clear();
                    m_alphas.clear();

                    for (size_t i = 0; i < n_samples; ++i) {
                        if (alpha_pool[i] > T(1e-5)) {
                            m_support_vectors.push_back(x_cache[i]);
                            m_support_vector_labels.push_back(y[i]);
                            m_alphas.push_back(alpha_pool[i]);
                        }
                    }

                    m_is_fitted = true;
                }

                /**
                 * @brief Classifies a sample matrix based on support vector geometric constraints.
                 */
                std::vector<size_t> predict(const Core::Matrix<T>& X) const {
                    if (!m_is_fitted) throw std::runtime_error("SVM Error: Fit loop execution required prior to calling predict.");

                    size_t n_samples = X.rows();
                    size_t n_features = X.cols();
                    std::vector<size_t> predictions(n_samples);

                    for (size_t i = 0; i < n_samples; ++i) {
                        std::vector<T> row(n_features);
                        for (size_t j = 0; j < n_features; ++j) {
                            row[j] = X(i, j);
                        }
                        predictions[i] = (compute_functional_margin(row) >= T(0.0)) ? 1 : 0;
                    }

                    return predictions;
                }
            };

        }
    }
}