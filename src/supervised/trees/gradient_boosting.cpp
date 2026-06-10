// src/supervised/trees/gradient_boosting.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Supervised {

        template <typename T>
        GradientBoostingClassifier<T>::GradientBoostingClassifier(size_t n_estimators, T learning_rate, size_t max_depth)
            : m_n_estimators(n_estimators), m_lr(learning_rate), m_max_depth(max_depth), m_initial_guess(0) {}

        template <typename T>
        void GradientBoostingClassifier<T>::fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) {
            size_t m = X.rows();
            m_trees.clear();
            m_trees.reserve(m_n_estimators);

            // 1. Initial baseline log-odds prediction computation
            T count_class_1 = 0;
            for (size_t i = 0; i < m; ++i) {
                if (y(i, 0) == T(1)) count_class_1 += T(1);
            }
            T p_initial = count_class_1 / static_cast<T>(m);
            if (p_initial <= T(0)) p_initial = T(0.01);
            if (p_initial >= T(1)) p_initial = T(0.99);
            
            m_initial_guess = std::log(p_initial / (T(1) - p_initial));

            // Setup continuous tracking array for running log-odds
            std::vector<T> current_log_odds(m, m_initial_guess);
            Core::Matrix<T> residuals(m, 1);

            for (size_t t = 0; t < m_n_estimators; ++t) {
                // 2. Compute Pseudo-Residuals (negative gradients): r_i = y_i - p_i
                for (size_t i = 0; i < m; ++i) {
                    T p = sigmoid(current_log_odds[i]);
                    residuals(i, 0) = y(i, 0) - p;
                }

                // 3. Fit a decision tree to step in the direction of the residuals
                DecisionTreeClassifier<T> residual_tree(m_max_depth, 2,true);
                residual_tree.fit(X, residuals);

                // 4. Update the running log-odds predictions using shrinkage (learning rate)
                Core::Matrix<T> tree_preds = residual_tree.predict(X);
                for (size_t i = 0; i < m; ++i) {
                    current_log_odds[i] += m_lr * tree_preds(i, 0);
                }

                m_trees.push_back(residual_tree);
            }
        }

        template <typename T>
        Core::Matrix<T> GradientBoostingClassifier<T>::predict_proba(const Core::Matrix<T>& X) const {
            size_t m = X.rows();
            Core::Matrix<T> probas(m, 1);
            std::vector<T> log_odds(m, m_initial_guess);

            // Accumulate residual predictions scaled by the learning rate
            for (const auto& tree : m_trees) {
                Core::Matrix<T> tree_preds = tree.predict(X);
                for (size_t i = 0; i < m; ++i) {
                    log_odds[i] += m_lr * tree_preds(i, 0);
                }
            }

            for (size_t i = 0; i < m; ++i) {
                probas(i, 0) = sigmoid(log_odds[i]);
            }
            return probas;
        }

        template <typename T>
        Core::Matrix<T> GradientBoostingClassifier<T>::predict(const Core::Matrix<T>& X) const {
            Core::Matrix<T> probas = predict_proba(X);
            Core::Matrix<T> classes(probas.rows(), 1);

            for (size_t i = 0; i < probas.rows(); ++i) {
                classes(i, 0) = (probas(i, 0) >= T(0.5)) ? T(1) : T(0);
            }
            return classes;
        }

    }
}