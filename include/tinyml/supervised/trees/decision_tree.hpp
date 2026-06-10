#pragma once
#include "tinyml/supervised/base.hpp"
#include <memory>
#include <vector>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class DecisionTreeClassifier : public BasePredictor<T> {
        private:
            struct Node {
                bool is_leaf = false;
                T value = T(0);             // Predicted class if it's a leaf node
                size_t feature_idx = 0;     // Feature index used for splitting
                T threshold = T(0);         // Threshold boundary value for split
                std::shared_ptr<Node> left = nullptr;
                std::shared_ptr<Node> right = nullptr;
            };

            std::shared_ptr<Node> m_root = nullptr;
            
            size_t m_max_depth;
            size_t m_min_samples_split;
            bool m_is_regression;
            
            // Internal recursive compilation builders
            std::shared_ptr<Node> build_tree(const Core::Matrix<T>& X, const Core::Matrix<T>& y, size_t current_depth);
            T calculate_gini(const Core::Matrix<T>& y) const;
            
            // Helper to split a matrix based on a feature boundary
            void split_dataset(const Core::Matrix<T>& X, size_t feature_idx, T threshold,
                               std::vector<size_t>& left_indices, std::vector<size_t>& right_indices) const;

            T predict_row(const std::shared_ptr<Node>& node, const Core::Matrix<T>& X, size_t row) const;

        public:
            DecisionTreeClassifier(size_t max_depth = 5, size_t min_samples_split = 2,bool is_regression = false);

            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override;
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override;
        };

    }
}