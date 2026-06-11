#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class PCA {
        private:
            size_t m_components;
            Core::Matrix<T> m_components_matrix; // Stores the principal component eigenvectors
            std::vector<T> m_means;              // Column means for centering
            bool m_is_fitted = false;

        public:
            PCA(size_t n_components = 2) : m_components(n_components) {}

            /**
             * @brief Fits the PCA model by calculating the principal components of X
             */
            void fit(const Core::Matrix<T>& X, size_t max_iters = 100, T tol = 1e-6) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                if (m_components > num_features) {
                    throw std::runtime_error("PCA Error: Target components cannot exceed original feature dimensions.");
                }

                // 1. Compute column-wise means and center the data
                m_means.assign(num_features, 0);
                Core::Matrix<T> X_centered(num_samples, num_features);
                
                for (size_t j = 0; j < num_features; ++j) {
                    T sum = 0;
                    for (size_t i = 0; i < num_samples; ++i) {
                        sum += X(i, j);
                    }
                    m_means[j] = sum / static_cast<T>(num_samples);
                    
                    for (size_t i = 0; i < num_samples; ++i) {
                        X_centered(i, j) = X(i, j) - m_means[j];
                    }
                }

                // 2. Compute Covariance Matrix: C = (X_centered^T * X_centered) / (N - 1)
                Core::Matrix<T> cov(num_features, num_features);
                T denominator = static_cast<T>(num_samples - 1);
                if (denominator <= 0) denominator = 1;

                for (size_t i = 0; i < num_features; ++i) {
                    for (size_t j = 0; j < num_features; ++j) {
                        T sum = 0;
                        for (size_t k = 0; k < num_samples; ++k) {
                            sum += X_centered(k, i) * X_centered(k, j);
                        }
                        cov(i, j) = sum / denominator;
                    }
                }

                // 3. Power Iteration to extract the top Principal Components
                m_components_matrix = Core::Matrix<T>(m_components, num_features);

                for (size_t comp = 0; comp < m_components; ++comp) {
                    // Initialize random vector estimate
                    std::vector<T> b_k(num_features, 1.0f); 

                    for (size_t iter = 0; iter < max_iters; ++iter) {
                        std::vector<T> b_k1(num_features, 0.0f);
                        T norm = 0;

                        // Multiply Covariance Matrix by b_k vector
                        for (size_t i = 0; i < num_features; ++i) {
                            for (size_t j = 0; j < num_features; ++j) {
                                b_k1[i] += cov(i, j) * b_k[j];
                            }
                            norm += b_k1[i] * b_k1[i];
                        }
                        norm = std::sqrt(norm);
                        if (norm < 1e-9) norm = 1;

                        // Normalize vector step
                        T change = 0;
                        for (size_t i = 0; i < num_features; ++i) {
                            T next_val = b_k1[i] / norm;
                            change += std::abs(next_val - b_k[i]);
                            b_k[i] = next_val;
                        }

                        if (change < tol) break;
                    }

                    // Save the calculated principal vector component row
                    for (size_t j = 0; j < num_features; ++j) {
                        m_components_matrix(comp, j) = b_k[j];
                    }

                    // Deflate the Covariance matrix to remove this component's variance
                    // cov = cov - eigenvalue * (b_k * b_k^T)
                    // Let's approximate Rayleigh quotient eigenvalue
                    T eigenvalue = 0;
                    std::vector<T> cov_bk(num_features, 0);
                    for (size_t i = 0; i < num_features; ++i) {
                        for (size_t j = 0; j < num_features; ++j) {
                            cov_bk[i] += cov(i, j) * b_k[j];
                        }
                        eigenvalue += b_k[i] * cov_bk[i];
                    }

                    for (size_t i = 0; i < num_features; ++i) {
                        for (size_t j = 0; j < num_features; ++j) {
                            cov(i, j) -= eigenvalue * b_k[i] * b_k[j];
                        }
                    }
                }

                m_is_fitted = true;
            }

            /**
             * @brief Transform high-dimensional data points down onto the principal components space
             */
            Core::Matrix<T> transform(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) throw std::runtime_error("PCA Error: Model must be fitted first.");

                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                // Center incoming vectors using fitted historical means
                Core::Matrix<T> X_centered(num_samples, num_features);
                for (size_t j = 0; j < num_features; ++j) {
                    for (size_t i = 0; i < num_samples; ++i) {
                        X_centered(i, j) = X(i, j) - m_means[j];
                    }
                }

                // Project centered matrix by Multiplying it with Transposed Components Matrix
                // Out = X_centered * Components^T
                Core::Matrix<T> projected(num_samples, m_components);
                for (size_t i = 0; i < num_samples; ++i) {
                    for (size_t c = 0; c < m_components; ++c) {
                        T sum = 0;
                        for (size_t j = 0; j < num_features; ++j) {
                            sum += X_centered(i, j) * m_components_matrix(c, j);
                        }
                        projected(i, c) = sum;
                    }
                }

                return projected;
            }

            const Core::Matrix<T>& get_components() const { return m_components_matrix; }
        };

    }
}