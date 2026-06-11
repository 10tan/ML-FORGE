#pragma once
#include "tinyml/core/matrix.hpp"
#include "tinyml/supervised/base.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <map>
#include <algorithm>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class DecisionTreeClassifier : public BasePredictor<T> {
        private:
            struct Node {
                bool is_leaf = false;
                size_t feature_idx = 0;
                T threshold = T(0.0);
                T value = T(0.0); // Predicted class label for leaf nodes
                std::shared_ptr<Node> left;
                std::shared_ptr<Node> right;
            };

            size_t m_max_depth;
            size_t m_min_samples_split;
            bool m_is_regressor; // Keeps flag layout for future regression support expansion
            std::shared_ptr<Node> m_root;

            /**
             * @brief Computes Gini Impurity for a label slice matrix.
             */
            T calculate_gini(const Core::Matrix<T>& y) const {
                size_t num_samples = y.rows(); 
                if (num_samples == 0) return T(0.0);

                std::map<size_t, size_t> counts;
                for (size_t i = 0; i < num_samples; ++i) {
                    counts[static_cast<size_t>(y(i, 0))]++;
                }

                T impurity = T(1.0);
                for (const auto& pair : counts) { // 🎯 Fixed: Changed from char to auto
                    T p = static_cast<T>(pair.second) / static_cast<T>(num_samples);
                    impurity -= p * p;
                }
                return impurity;
            }

            /**
             * @brief Computes Variance reduction metrics for node splits.
             */
            T calculate_variance(const Core::Matrix<T>& y) const {
                size_t num_samples = y.rows(); // 🎯 Fixed: Changed from y.size()
                if (num_samples == 0) return T(0.0);

                T sum = T(0.0);
                for (size_t i = 0; i < num_samples; ++i) sum += y(i, 0);
                T mean = sum / static_cast<T>(num_samples);

                T variance = T(0.0);
                for (size_t i = 0; i < num_samples; ++i) {
                    T diff = y(i, 0) - mean;
                    variance += diff * diff;
                }
                return variance / static_cast<T>(num_samples);
            }

            /**
             * @brief Recursive builder logic that greedily searches feature split regions.
             */
            std::shared_ptr<Node> build_tree(const Core::Matrix<T>& X, const Core::Matrix<T>& y, size_t depth) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                auto node = std::make_shared<Node>();

                // Base Case: Check stopping conditions (depth bounds, sample minimums, purity)
                if (depth >= m_max_depth || num_samples < m_min_samples_split || 
                    (!m_is_regressor && calculate_gini(y) == T(0.0))) {
                    node->is_leaf = true;
                    node->value = compute_leaf_value(y);
                    return node;
                }

                size_t best_feat = 0;
                T best_thresh = T(0.0);
                T best_score = m_is_regressor ? std::numeric_limits<T>::max() : std::numeric_limits<T>::max();
                bool split_found = false;

                // Loop through dimensions to search for optimization boundaries
                for (size_t f = 0; f < num_features; ++f) {
                    std::vector<T> feature_values(num_samples);
                    for (size_t i = 0; i < num_samples; ++i) feature_values[i] = X(i, f);
                    std::sort(feature_values.begin(), feature_values.end());

                    for (size_t i = 1; i < num_samples; ++i) {
                        T thresh = (feature_values[i - 1] + feature_values[i]) / T(2.0);
                        
                        // Partition matrices
                        std::vector<size_t> left_indices, right_indices;
                        for (size_t r = 0; r < num_samples; ++r) {
                            if (X(r, f) <= thresh) left_indices.push_back(r);
                            else                   right_indices.push_back(r);
                        }

                        if (left_indices.empty() || right_indices.empty()) continue;

                        T score = T(0.0);
                        if (!m_is_regressor) {
                            // Calculate combined Gini impurity split scores
                            Core::Matrix<T> y_left(left_indices.size(), 1);
                            Core::Matrix<T> y_right(right_indices.size(), 1);
                            for (size_t idx = 0; idx < left_indices.size(); ++idx) y_left(idx, 0) = y(left_indices[idx], 0);
                            for (size_t idx = 0; idx < right_indices.size(); ++idx) y_right(idx, 0) = y(right_indices[idx], 0);

                            T gini_l = calculate_gini(y_left);
                            T gini_r = calculate_gini(y_right);
                            score = (static_cast<T>(left_indices.size()) / num_samples) * gini_l +
                                    (static_cast<T>(right_indices.size()) / num_samples) * gini_r;
                        }

                        if (score < best_score) {
                            best_score = score;
                            best_feat = f;
                            best_thresh = thresh;
                            split_found = true;
                        }
                    }
                }

                if (!split_found) {
                    node->is_leaf = true;
                    node->value = compute_leaf_value(y);
                    return node;
                }

                // Partition data to recurse downstream branches
                std::vector<size_t> left_idxs, right_idxs;
                for (size_t r = 0; r < num_samples; ++r) {
                    if (X(r, best_feat) <= best_thresh) left_idxs.push_back(r);
                    else                                right_idxs.push_back(r);
                }

                Core::Matrix<T> X_left(left_idxs.size(), num_features);
                Core::Matrix<T> y_left(left_idxs.size(), 1);
                Core::Matrix<T> X_right(right_idxs.size(), num_features);
                Core::Matrix<T> y_right(right_idxs.size(), 1);

                for (size_t i = 0; i < left_idxs.size(); ++i) {
                    y_left(i, 0) = y(left_idxs[i], 0);
                    for (size_t j = 0; j < num_features; ++j) X_left(i, j) = X(left_idxs[i], j);
                }
                for (size_t i = 0; i < right_idxs.size(); ++i) {
                    y_right(i, 0) = y(right_idxs[i], 0);
                    for (size_t j = 0; j < num_features; ++j) X_right(i, j) = X(right_idxs[i], j);
                }

                node->feature_idx = best_feat;
                node->threshold = best_thresh;
                node->left = build_tree(X_left, y_left, depth + 1);
                node->right = build_tree(X_right, y_right, depth + 1);

                return node;
            }

            /**
             * @brief Computes majority mode value out of classification leaf nodes.
             */
            T compute_leaf_value(const Core::Matrix<T>& y) const {
                size_t num_samples = y.rows(); // 🎯 Fixed: Changed from y.size()
                std::map<size_t, size_t> counts;
                size_t majority_label = 0;
                size_t max_count = 0;

                for (size_t i = 0; i < num_samples; ++i) {
                    size_t label = static_cast<size_t>(y(i, 0));
                    counts[label]++;
                    if (counts[label] > max_count) {
                        max_count = counts[label];
                        majority_label = label;
                    }
                }
                return static_cast<T>(majority_label);
            }

            T predict_sample(std::shared_ptr<Node> node, const std::vector<T>& sample) const {
                if (node->is_leaf) return node->value;
                if (sample[node->feature_idx] <= node->threshold) {
                    return predict_sample(node->left, sample);
                }
                return predict_sample(node->right, sample);
            }

        public:
            DecisionTreeClassifier(size_t max_depth = 5, size_t min_samples_split = 2, bool is_regressor = false)
                : m_max_depth(max_depth), m_min_samples_split(min_samples_split), m_is_regressor(is_regressor), m_root(nullptr) {}

            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override {
                if (X.rows() == 0 || X.rows() != y.rows()) {
                    throw std::runtime_error("Tree Error: Data shapes must balance perfectly across configurations.");
                }
                m_root = build_tree(X, y, 0);
            }

            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override {
                if (!m_root) throw std::runtime_error("Tree Error: Fit execution missing before prediction updates.");
                
                size_t num_samples = X.rows();
                size_t num_features = X.cols();
                Core::Matrix<T> predictions(num_samples, 1);

                for (size_t i = 0; i < num_samples; ++i) {
                    std::vector<T> sample(num_features);
                    for (size_t j = 0; j < num_features; ++j) sample[j] = X(i, j);
                    predictions(i, 0) = predict_sample(m_root, sample); // 🎯 Fixed: Changed from [i, 0] to (i, 0)
                }
                return predictions;
            }
        };

    }
}