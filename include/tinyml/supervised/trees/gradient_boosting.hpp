#pragma once
#include "tinyml/supervised/base.hpp"
#include "tinyml/supervised/trees/decision_tree.hpp"
#include <vector>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class GradientBoostingClassifier : public BasePredictor<T> {
        private:
            size_t m_n_estimators; // Number of sequential boosting trees
            T m_lr;                // Learning rate / shrinkage step modifier
            size_t m_max_depth;    // Control limits for residual tree growth
            T m_initial_guess;     // Base log-odds value initialization log(p / (1-p))

            std::vector<DecisionTreeClassifier<T>> m_trees;

            // Internal helper to convert log-odds to clean probability spaces
            T sigmoid(T log_odds) const { return T(1) / (T(1) + std::exp(-log_odds)); }

        public:
            GradientBoostingClassifier(size_t n_estimators = 100, T learning_rate = T(0.1), size_t max_depth = 3);

            // Interface implementations
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override;
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override;
            
            // Output predictive raw probabilities
            Core::Matrix<T> predict_proba(const Core::Matrix<T>& X) const;
        };

    }
}