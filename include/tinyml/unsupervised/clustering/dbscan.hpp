#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <queue>

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class DBSCAN {
        private:
            T m_eps;               // Scan neighborhood radius
            size_t m_min_samples;  // Core point density threshold
            std::vector<int> m_labels; // Cluster assignments (-1 for noise/outliers)
            bool m_is_fitted = false;

            // Helper: Compute Euclidean distance between two rows
            T compute_dist(const Core::Matrix<T>& X, size_t idx1, size_t idx2) const {
                T dist_sum = 0;
                for (size_t j = 0; j < X.cols(); ++j) {
                    T diff = X(idx1, j) - X(idx2, j);
                    dist_sum += diff * diff;
                }
                return std::sqrt(dist_sum);
            }

            // Helper: Find all sample indices inside the epsilon neighborhood of point_idx
            std::vector<size_t> region_query(const Core::Matrix<T>& X, size_t point_idx) const {
                std::vector<size_t> neighbors;
                for (size_t i = 0; i < X.rows(); ++i) {
                    if (compute_dist(X, point_idx, i) <= m_eps) {
                        neighbors.push_back(i);
                    }
                }
                return neighbors;
            }

        public:
            DBSCAN(T eps = T(0.5), size_t min_samples = 5) 
                : m_eps(eps), m_min_samples(min_samples) {}

            /**
             * @brief Fits density clusters onto the matrix workspace
             */
            void fit(const Core::Matrix<T>& X) {
                size_t num_samples = X.rows();
                m_labels.assign(num_samples, 0); // 0 means unvisited

                int cluster_id = 0;

                for (size_t i = 0; i < num_samples; ++i) {
                    if (m_labels[i] != 0) continue; // Skip already processed points

                    std::vector<size_t> neighbors = region_query(X, i);

                    if (neighbors.size() < m_min_samples) {
                        m_labels[i] = -1; // Initially flag as noise/outlier
                    } else {
                        cluster_id++; // Found a new core density group seed!
                        
                        // Expand the cluster using a breadth-first queue search
                        m_labels[i] = cluster_id;
                        std::queue<size_t> seeds;
                        for (size_t n : neighbors) {
                            if (n != i) seeds.push(n);
                        }

                        while (!seeds.empty()) {
                            size_t current_point = seeds.front();
                            seeds.pop();

                            if (m_labels[current_point] == -1) {
                                m_labels[current_point] = cluster_id; // Change noise to border point
                            }

                            if (m_labels[current_point] != 0) continue; // Skip visited points

                            m_labels[current_point] = cluster_id;
                            std::vector<size_t> current_neighbors = region_query(X, current_point);

                            if (current_neighbors.size() >= m_min_samples) {
                                for (size_t cn : current_neighbors) {
                                    seeds.push(cn);
                                }
                            }
                        }
                    }
                }

                m_is_fitted = true;
            }

            // Remap labels to standard sizes for visualizer compatibility (convert noise -1 to a clean category index)
            std::vector<size_t> get_cluster_labels() const {
                std::vector<size_t> visual_labels(m_labels.size());
                for (size_t i = 0; i < m_labels.size(); ++i) {
                    if (m_labels[i] <= 0) visual_labels[i] = 0; // Outliers cluster group 0
                    else visual_labels[i] = static_cast<size_t>(m_labels[i]);
                }
                return visual_labels;
            }

            const std::vector<int>& get_raw_labels() const { return m_labels; }
        };

    }
}