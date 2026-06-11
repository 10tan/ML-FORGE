#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class AgglomerativeClustering {
        private:
            size_t m_k;                // Target number of clusters
            std::vector<size_t> m_labels;
            bool m_is_fitted = false;

            // Helper: Compute Euclidean distance between two specific sample rows
            T compute_dist(const Core::Matrix<T>& X, size_t idx1, size_t idx2) const {
                T dist_sum = 0;
                for (size_t j = 0; j < X.cols(); ++j) {
                    T diff = X(idx1, j) - X(idx2, j);
                    dist_sum += diff * diff;
                }
                return std::sqrt(dist_sum);
            }

        public:
            AgglomerativeClustering(size_t n_clusters = 2) : m_k(n_clusters) {}

            /**
             * @brief Fits the bottom-up hierarchy and assigns cluster markers
             */
            void fit(const Core::Matrix<T>& X) {
                size_t num_samples = X.rows();
                if (num_samples < m_k || num_samples == 0) {
                    throw std::runtime_error("Hierarchical Error: Total samples cannot be fewer than target cluster count K.");
                }

                // 1. Initialize: Every single point begins its life inside its own distinct cluster ID
                m_labels.resize(num_samples);
                std::iota(m_labels.begin(), m_labels.end(), 0);

                // Precompute full distance matrix between all points
                Core::Matrix<T> dist_matrix(num_samples, num_samples);
                for (size_t i = 0; i < num_samples; ++i) {
                    for (size_t j = i + 1; j < num_samples; ++j) {
                        T d = compute_dist(X, i, j);
                        dist_matrix(i, j) = d;
                        dist_matrix(j, i) = d;
                    }
                }

                // Track current total unique active cluster categories
                size_t current_clusters = num_samples;

                // 2. Iterative Merge Loop: Fuse clusters until current_clusters == m_k
                while (current_clusters > m_k) {
                    T min_cluster_dist = std::numeric_limits<T>::max();
                    size_t cluster_to_merge_a = 0;
                    size_t cluster_to_merge_b = 0;

                    // Search for the two closest unique clusters using Single Linkage criteria
                    for (size_t i = 0; i < num_samples; ++i) {
                        for (size_t j = i + 1; j < num_samples; ++j) {
                            // Only look at points that do not already belong to the same cluster group
                            if (m_labels[i] != m_labels[j]) {
                                T current_pair_dist = dist_matrix(i, j);
                                if (current_pair_dist < min_cluster_dist) {
                                    min_cluster_dist = current_pair_dist;
                                    cluster_to_merge_a = m_labels[i];
                                    cluster_to_merge_b = m_labels[j];
                                }
                            }
                        }
                    }

                    // Standardize cluster identifier boundaries: Always merge into the lower cluster ID value
                    size_t target_cluster = std::min(cluster_to_merge_a, cluster_to_merge_b);
                    size_t obsolete_cluster = std::max(cluster_to_merge_a, cluster_to_merge_b);

                    // Reassign all instances of the obsolete cluster to our target cluster
                    for (size_t i = 0; i < num_samples; ++i) {
                        if (m_labels[i] == obsolete_cluster) {
                            m_labels[i] = target_cluster;
                        }
                    }

                    current_clusters--;
                }

                // 3. Compact labels array down into clean sequential integers (e.g., [0, 1, 2...])
                std::vector<size_t> unique_ids;
                for (size_t label : m_labels) {
                    if (std::find(unique_ids.begin(), unique_ids.end(), label) == unique_ids.end()) {
                        unique_ids.push_back(label);
                    }
                }

                for (size_t i = 0; i < num_samples; ++i) {
                    auto it = std::find(unique_ids.begin(), unique_ids.end(), m_labels[i]);
                    m_labels[i] = std::distance(unique_ids.begin(), it);
                }

                m_is_fitted = true;
            }

            const std::vector<size_t>& get_labels() const {
                if (!m_is_fitted) throw std::runtime_error("Hierarchical Error: Model must be fitted first.");
                return m_labels;
            }
        };

    }
}