// src/supervised/trees/random_forest.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
        RandomForestClassifier<T>::RandomForestClassifier(size_t n_estimators, size_t max_depth, 
                                                           size_t min_samples_split, T max_features_ratio)
            : m_n_estimators(n_estimators), m_max_depth(max_depth), 
              m_min_samples_split(min_samples_split), m_max_features_ratio(max_features_ratio) {
            std::random_device rd;
            m_rng.seed(rd());
        }

        template <typename T>
        void RandomForestClassifier<T>::bootstrap_sample(const Core::Matrix<T>& X, const Core::Matrix<T>& y,
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

        template <typename T>
        void RandomForestClassifier<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
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

        template <typename T>
        Core::Matrix<T> RandomForestClassifier<T>::predict(const Core::Matrix<T>& X) const {
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

    }
}