#pragma once
#include "tinyml/core/matrix.hpp"
#include "tinyml/supervised/base.hpp"
#include "tinyml/supervised/trees/decision_tree.hpp"
#include <vector>
#include <random>
#include <stdexcept>

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class RandomForestClassifier : public BasePredictor<T> {
        private:
            size_t m_n_estimators;
            size_t m_max_depth;
            size_t m_min_samples_split;
            T m_max_features_ratio;
            
            std::vector<DecisionTreeClassifier<T>> m_trees;
            mutable std::mt19937 m_rng;

            /**
             * @brief Generates a bootstrap sample matrix by sampling rows with replacement.
             */
            void bootstrap_sample(const Core::Matrix<T>& X, const Core::Matrix<T>& y,
                                  Core::Matrix<T>& X_boot, Core::Matrix<T>& y_boot) const {
                size_t m = X.rows();
                size_t n = X.cols();
                std::uniform_int_distribution<size_t> dist(0, m - 1);

                for (size_t i = 0; i < m; ++i) {
                    size_t random_row_idx = dist(m_rng);
                    y_boot(i, 0) = y(random_row_idx, 0);
                    for (size_t j = 0; j < n; ++j) {
                        X_boot(i, j) = X(random_row_idx, j);
                    }
                }
            }

        public:
            RandomForestClassifier(size_t n_estimators = 10, size_t max_depth = 10, 
                                   size_t min_samples_split = 2, T max_features_ratio = T(1.0))
                : m_n_estimators(n_estimators), m_max_depth(max_depth), 
                  m_min_samples_split(min_samples_split), m_max_features_ratio(max_features_ratio) {
                std::random_device rd;
                m_rng.seed(rd());
            }

            /**
             * @brief Trains the ensemble by fitting independent decision trees on bootstrapped data.
             */
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override {
                m_trees.clear();
                m_trees.reserve(m_n_estimators);

                for (size_t i = 0; i < m_n_estimators; ++i) {
                    Core::Matrix<T> X_boot(X.rows(), X.cols());
                    Core::Matrix<T> y_boot(y.rows(), 1);
                    bootstrap_sample(X, y, X_boot, y_boot);

                    // Initialize a base tree estimator with depth constraints
                    DecisionTreeClassifier<T> tree(m_max_depth, m_min_samples_split, false);
                    tree.fit(X_boot, y_boot);
                    m_trees.push_back(tree);
                }
            }

            /**
             * @brief Aggregates tree predictions using a majority consensus loop.
             */
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override {
                size_t m = X.rows();
                Core::Matrix<T> final_predictions(m, 1);

                // Collect predictions across all estimators
                std::vector<Core::Matrix<T>> forest_votes;
                forest_votes.reserve(m_n_estimators);
                for (const auto& tree : m_trees) {
                    forest_votes.push_back(tree.predict(X));
                }

                // Aggregate votes using a majority consensus loop
                for (size_t r = 0; r < m; ++r) {
                    std::vector<size_t> class_counts;
                    size_t majority_class = 0;
                    size_t max_votes = 0;

                    for (size_t t = 0; t < m_n_estimators; ++t) {
                        size_t vote = static_cast<size_t>(forest_votes[t](r, 0));
                        if (vote >= class_counts.size()) class_counts.resize(vote + 1, 0);
                        class_counts[vote]++;
                        
                        if (class_counts[vote] > max_votes) {
                            max_votes = class_counts[vote];
                            majority_class = vote;
                        }
                    }
                    final_predictions(r, 0) = static_cast<T>(majority_class);
                }

                return final_predictions;
            }
        };

    }
}