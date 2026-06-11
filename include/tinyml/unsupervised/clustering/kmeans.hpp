#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class KMeans {
        private:
            size_t m_k;          // Number of target cluster groups
            size_t m_max_iters;  // Convergence loop protection ceiling
            Core::Matrix<T> m_centroids;
            bool m_is_fitted = false;

            // Helper: Computes squared Euclidean distance between a data row and a centroid row
            T compute_distance(const Core::Matrix<T>& X, size_t row_idx, const Core::Matrix<T>& centroids, size_t c_idx) const {
                T distance_sum = 0;
                for (size_t j = 0; j < X.cols(); ++j) {
                    T diff = X(row_idx, j) - centroids(c_idx, j);
                    distance_sum += diff * diff;
                }
                return std::sqrt(distance_sum);
            }

        public:
            KMeans(size_t k = 3, size_t max_iters = 100) 
                : m_k(k), m_max_iters(max_iters) {}

            /**
             * @brief Fits the cluster centroids to the input feature matrix dataset
             */
            void fit(const Core::Matrix<T>& X, unsigned int seed = 42) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                if (num_samples < m_k) {
                    throw std::runtime_error("Clustering Error: Number of samples cannot be less than cluster parameter K.");
                }

                // 1. Initialize centroids randomly by picking sample rows (Forgy Method)
                m_centroids = Core::Matrix<T>(m_k, num_features);
                std::vector<size_t> sample_indices(num_samples);
                for (size_t i = 0; i < num_samples; ++i) sample_indices[i] = i;
                
                std::mt19937 rng(seed);
                std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

                for (size_t i = 0; i < m_k; ++i) {
                    for (size_t j = 0; j < num_features; ++j) {
                        m_centroids(i, j) = X(sample_indices[i], j);
                    }
                }

                std::vector<size_t> assignments(num_samples, 0);

                // 2. Main Expectation-Maximization Iterative Loop
                for (size_t iter = 0; iter < m_max_iters; ++iter) {
                    bool changed = false;

                    // --- Expectation Step: Assign each sample point to its closest centroid ---
                    for (size_t i = 0; i < num_samples; ++i) {
                        size_t best_cluster = 0;
                        T min_dist = std::numeric_limits<T>::max();

                        for (size_t c = 0; c < m_k; ++c) {
                            T dist = compute_distance(X, i, m_centroids, c);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_cluster = c;
                            }
                        }

                        if (assignments[i] != best_cluster) {
                            assignments[i] = best_cluster;
                            changed = true;
                        }
                    }

                    // If no sample assignments shifted, we have converged completely!
                    if (!changed) break;

                    // --- Maximization Step: Recompute centroid positions to the mean of their points ---
                    Core::Matrix<T> new_centroids(m_k, num_features);
                    std::vector<size_t> cluster_counts(m_k, 0);

                    for (size_t i = 0; i < num_samples; ++i) {
                        size_t c = assignments[i];
                        cluster_counts[c]++;
                        for (size_t j = 0; j < num_features; ++j) {
                            new_centroids(c, j) += X(i, j);
                        }
                    }

                    for (size_t c = 0; c < m_k; ++c) {
                        if (cluster_counts[c] > 0) {
                            for (size_t j = 0; j < num_features; ++j) {
                                m_centroids(c, j) = new_centroids(c, j) / static_cast<T>(cluster_counts[c]);
                            }
                        }
                    }
                }

                m_is_fitted = true;
            }

            /**
             * @brief Predicts the closest cluster assignment indices for a new set of data points
             */
            std::vector<size_t> predict(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) throw std::runtime_error("Clustering Error: Model must be fitted before clustering.");

                std::vector<size_t> assignments(X.rows());
                for (size_t i = 0; i < X.rows(); ++i) {
                    size_t best_cluster = 0;
                    T min_dist = std::numeric_limits<T>::max();

                    for (size_t c = 0; c < m_k; ++c) {
                        T dist = compute_distance(X, i, m_centroids, c);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster = c;
                        }
                    }
                    assignments[i] = best_cluster;
                }
                return assignments;
            }

            // Accessor to inspect computed centroids layout
            const Core::Matrix<T>& get_centroids() const { return m_centroids; }
        };

    }
}