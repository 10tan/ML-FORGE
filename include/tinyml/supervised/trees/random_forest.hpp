#pragma once
#include "tinyml/supervised/base.hpp"
#include "tinyml/supervised/trees/decision_tree.hpp"
#include <vector>
#include <random>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class RandomForestClassifier : public BasePredictor<T> {
        private:
            size_t m_n_estimators;       // Number of trees in the forest
            size_t m_max_depth;           // Max depth allowed per individual tree
            size_t m_min_samples_split;   // Min samples required to split a node
            T m_max_features_ratio;      // Fraction of total features to evaluate per split (e.g., 0.7)

            std::vector<DecisionTreeClassifier<T>> m_trees;
            mutable std::mt19937 m_rng;  // Random number engine for generating bootstraps

            // Helper to generate a bootstrap sample dataset (sampling rows with replacement)
            void bootstrap_sample(const Core::Matrix<T>& X, const Core::Matrix<T>& y,
                                  Core::Matrix<T>& X_boot, Core::Matrix<T>& y_boot) const;

        public:
            RandomForestClassifier(size_t n_estimators = 10, size_t max_depth = 5, 
                                   size_t min_samples_split = 2, T max_features_ratio = T(0.7));

            // Interface implementations
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override;
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override;
        };

    }
}