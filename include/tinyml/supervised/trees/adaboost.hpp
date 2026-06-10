#pragma once
#include "tinyml/supervised/base.hpp"
#include "tinyml/supervised/trees/decision_tree.hpp"
#include <vector>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class AdaBoostClassifier : public BasePredictor<T> {
        private:
            size_t m_n_estimators; // Number of sequential boosting rounds
            T m_learning_rate;     // Shrinkage parameter for estimator weights

            std::vector<DecisionTreeClassifier<T>> m_estimators;
            std::vector<T> m_estimator_weights; // Alpha weights for voting power

        public:
            AdaBoostClassifier(size_t n_estimators = 50, T learning_rate = T(1.0));

            // Interface implementations
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override;
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override;
        };

    }
}