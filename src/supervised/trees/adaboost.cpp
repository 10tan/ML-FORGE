// src/supervised/trees/adaboost.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
        AdaBoostClassifier<T>::AdaBoostClassifier(size_t n_estimators, T learning_rate)
            : m_n_estimators(n_estimators), m_learning_rate(learning_rate) {}

        template <typename T>
        void AdaBoostClassifier<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
            size_t m = X.rows();
            m_estimators.clear();
            m_estimator_weights.clear();

            // 1. Initialize uniform sample weights: w_i = 1 / m
            std::vector<T> sample_weights(m, T(1) / static_cast<T>(m));

            for (size_t t = 0; t < m_n_estimators; ++t) {
                // For a robust implementation from scratch, we fit a shallow decision stump
                DecisionTreeClassifier<T> stump(1, 2, false); 
                stump.fit(X, y);

                Core::Matrix<T> preds = stump.predict(X);

                // 2. Compute the weighted error rate (epsilon)
                T weighted_error = 0;
                for (size_t i = 0; i < m; ++i) {
                    if (std::abs(preds(i, 0) - y(i, 0)) > T(1e-5)) {
                        weighted_error += sample_weights[i];
                    }
                }

                // Avoid division by zero or negative log scenarios
                if (weighted_error >= T(0.5) || weighted_error <= T(1e-7)) {
                    if (t == 0) { // Keep at least one valid base estimator
                        m_estimators.push_back(stump);
                        m_estimator_weights.push_back(T(1.0));
                    }
                    break; 
                }

                // 3. Compute estimator voting power (Alpha)
                T alpha = m_learning_rate * T(0.5) * std::log((T(1) - weighted_error) / weighted_error);

                // 4. Update row distribution weights exponentially
                T weight_sum = 0;
                for (size_t i = 0; i < m; ++i) {
                    T prediction_sign = (std::abs(preds(i, 0) - y(i, 0)) < T(1e-5)) ? T(1) : T(-1);
                    sample_weights[i] *= std::exp(-alpha * prediction_sign);
                    weight_sum += sample_weights[i];
                }

                // Normalize weights so they sum to 1
                for (size_t i = 0; i < m; ++i) {
                    sample_weights[i] /= weight_sum;
                }

                m_estimators.push_back(stump);
                m_estimator_weights.push_back(alpha);
            }
        }

        template <typename T>
        Core::Matrix<T> AdaBoostClassifier<T>::predict(const Core::Matrix<T>& X) const {
            size_t m = X.rows();
            Core::Matrix<T> final_predictions(m, 1);

            for (size_t i = 0; i < m; ++i) {
                std::vector<T> class_scores;
                for (size_t t = 0; t < m_estimators.size(); ++t) {
                    size_t vote = static_cast<size_t>(m_estimators[t].predict(X)(i, 0));
                    if (vote >= class_scores.size()) class_scores.resize(vote + 1, T(0));
                    class_scores[vote] += m_estimator_weights[t]; // Accumulate alpha scores
                }

                size_t high_score_class = 0;
                T max_score = -1.0f;
                for (size_t c = 0; c < class_scores.size(); ++c) {
                    if (class_scores[c] > max_score) {
                        max_score = class_scores[c];
                        high_score_class = c;
                    }
                }
                final_predictions(i, 0) = static_cast<T>(high_score_class);
            }

            return final_predictions;
        }

    }
}