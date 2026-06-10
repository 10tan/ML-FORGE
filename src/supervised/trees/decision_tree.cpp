// src/supervised/trees/decision_tree.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
DecisionTreeClassifier<T>::DecisionTreeClassifier(size_t max_depth, size_t min_samples_split, bool is_regression)
    : m_max_depth(max_depth), m_min_samples_split(min_samples_split), m_is_regression(is_regression) {}

        template <typename T>
        T DecisionTreeClassifier<T>::calculate_gini(const Core::Matrix<T>& y) const {
            if (y.size() == 0) return T(0);
            
            // Count occurrences of each unique integer label class
            // (Assuming standard classification formats: e.g. 0.0, 1.0, 2.0)
            std::vector<size_t> counts;
            for (size_t i = 0; i < y.size(); ++i) {
                size_t label = static_cast<size_t>(y.data()[i]);
                if (label >= counts.size()) counts.resize(label + 1, 0);
                counts[label]++;
            }

            T sum_squares = 0;
            T total = static_cast<T>(y.size());
            for (size_t count : counts) {
                T p = static_cast<T>(count) / total;
                sum_squares += p * p;
            }
            return T(1) - sum_squares;
        }

        template <typename T>
        void DecisionTreeClassifier<T>::split_dataset(const Core::Matrix<T>& X, 
                                             size_t feature_idx, T threshold,
                                             std::vector<size_t>& left_indices, std::vector<size_t>& right_indices) const {
    left_indices.clear();
    right_indices.clear();
            for (size_t r = 0; r < X.rows(); ++r) {
                if (X(r, feature_idx) <= threshold) {
                    left_indices.push_back(r);
                } else {
                    right_indices.push_back(r);
                }
            }
        }

        template <typename T>
std::shared_ptr<typename DecisionTreeClassifier<T>::Node> 
DecisionTreeClassifier<T>::build_tree(const Core::Matrix<T>& X, const Core::Matrix<T>& y, size_t current_depth) {
    size_t num_samples = X.rows();
    size_t num_features = X.cols();

    auto node = std::make_shared<Node>();

    // Calculate leaf value assignment up front
    if (m_is_regression) {
        T sum = 0;
        for (size_t i = 0; i < y.size(); ++i) sum += y.data()[i];
        node->value = (y.size() > 0) ? (sum / static_cast<T>(y.size())) : T(0);
    } else {
        std::vector<size_t> label_counts;
        size_t majority_class = 0;
        size_t max_count = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            size_t val = static_cast<size_t>(y.data()[i]);
            if (val >= label_counts.size()) label_counts.resize(val + 1, 0);
            label_counts[val]++;
            if (label_counts[val] > max_count) {
                max_count = label_counts[val];
                majority_class = val;
            }
        }
        node->value = static_cast<T>(majority_class);
    }

    // Base cases check: Stop splitting if criteria met
    // (If in regression mode, stop splitting if variance drops to 0)
    T current_impurity = 0;
    if (m_is_regression) {
        T mean = node->value;
        for (size_t i = 0; i < y.size(); ++i) {
            T diff = y.data()[i] - mean;
            current_impurity += diff * diff;
        }
    } else {
        current_impurity = calculate_gini(y);
    }

    if (current_depth >= m_max_depth || num_samples < m_min_samples_split || current_impurity == 0) {
        node->is_leaf = true;
        return node;
    }

    // Greedy Search Loop for the Best Split Boundary
    T best_impurity_score = 99999.0f;
    size_t best_feature = 0;
    T best_threshold = 0;
    std::vector<size_t> best_left_idxs, best_right_idxs;

    for (size_t f = 0; f < num_features; ++f) {
        for (size_t r = 0; r < num_samples; ++r) {
            T threshold = X(r, f);
            std::vector<size_t> left_idxs, right_idxs;
            split_dataset(X, f, threshold, left_idxs, right_idxs);

            if (left_idxs.empty() || right_idxs.empty()) continue;

            // Slice child targets
            Core::Matrix<T> y_left(left_idxs.size(), 1);
            for (size_t i = 0; i < left_idxs.size(); ++i) y_left(i, 0) = y(left_idxs[i], 0);

            Core::Matrix<T> y_right(right_idxs.size(), 1);
            for (size_t i = 0; i < right_idxs.size(); ++i) y_right(i, 0) = y(right_idxs[i], 0);

            T total_impurity = 0;
            if (m_is_regression) {
                // Calculate Variance/MSE split cost: sum((y_left - mean_left)^2) + sum((y_right - mean_right)^2)
                T mean_l = 0, mean_r = 0;
                for (size_t i = 0; i < y_left.size(); ++i) mean_l += y_left.data()[i];
                if (y_left.size() > 0) mean_l /= static_cast<T>(y_left.size());

                for (size_t i = 0; i < y_right.size(); ++i) mean_r += y_right.data()[i];
                if (y_right.size() > 0) mean_r /= static_cast<T>(y_right.size());

                T var_l = 0, var_r = 0;
                for (size_t i = 0; i < y_left.size(); ++i) { T d = y_left.data()[i] - mean_l; var_l += d * d; }
                for (size_t i = 0; i < y_right.size(); ++i) { T d = y_right.data()[i] - mean_r; var_r += d * d; }
                
                total_impurity = var_l + var_r;
            } else {
                // Weighted Gini cost calculation for standard classification
                T weight_l = static_cast<T>(left_idxs.size()) / static_cast<T>(num_samples);
                T weight_r = static_cast<T>(right_idxs.size()) / static_cast<T>(num_samples);
                total_impurity = (weight_l * calculate_gini(y_left)) + (weight_r * calculate_gini(y_right));
            }

            if (total_impurity < best_impurity_score) {
                best_impurity_score = total_impurity;
                best_feature = f;
                best_threshold = threshold;
                best_left_idxs = left_idxs;
                best_right_idxs = right_idxs;
            }
        }
    }

    if (best_impurity_score == 99999.0f) {
        node->is_leaf = true;
        return node;
    }

    // Assign parameters and split recursively
    node->feature_idx = best_feature;
    node->threshold = best_threshold;

    Core::Matrix<T> X_left(best_left_idxs.size(), num_features);
    Core::Matrix<T> y_left(best_left_idxs.size(), 1);
    for (size_t i = 0; i < best_left_idxs.size(); ++i) {
        y_left(i, 0) = y(best_left_idxs[i], 0);
        for (size_t j = 0; j < num_features; ++j) X_left(i, j) = X(best_left_idxs[i], j);
    }

    Core::Matrix<T> X_right(best_right_idxs.size(), num_features);
    Core::Matrix<T> y_right(best_right_idxs.size(), 1);
    for (size_t i = 0; i < best_right_idxs.size(); ++i) {
        y_right(i, 0) = y(best_right_idxs[i], 0);
        for (size_t j = 0; j < num_features; ++j) X_right(i, j) = X(best_right_idxs[i], j);
    }

    node->left = build_tree(X_left, y_left, current_depth + 1);
    node->right = build_tree(X_right, y_right, current_depth + 1);
    return node;
}

        template <typename T>
        void DecisionTreeClassifier<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
            m_root = build_tree(X, y, 0);
        }

        template <typename T>
        T DecisionTreeClassifier<T>::predict_row(const std::shared_ptr<Node>& node, const Core::Matrix<T>& X, size_t row) const {
            if (node->is_leaf) return node->value;
            if (X(row, node->feature_idx) <= node->threshold) {
                return predict_row(node->left, X, row);
            }
            return predict_row(node->right, X, row);
        }

        template <typename T>
        Core::Matrix<T> DecisionTreeClassifier<T>::predict(const Core::Matrix<T>& X) const {
            Core::Matrix<T> preds(X.rows(), 1);
            for (size_t i = 0; i < X.rows(); ++i) {
                preds(i, 0) = predict_row(m_root, X, i);
            }
            return preds;
        }

    }
}