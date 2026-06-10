#pragma once
#include "tinyml/supervised/base.hpp"
#include "tinyml/supervised/nn/activations/sigmoid.hpp"

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class LogisticRegression : public BasePredictor<T> {
        private:
            Core::Matrix<T> m_weights;
            T m_bias;
            T m_lr;
            size_t m_iters;

        public:
            LogisticRegression(T learning_rate = 0.01, size_t iterations = 1000);

            // Interface implementations
            void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) override;
            Core::Matrix<T> predict(const Core::Matrix<T>& X) const override;
            
            // Helper method to output raw probabilities instead of hard 0 or 1 boundaries
            Core::Matrix<T> predict_proba(const Core::Matrix<T>& X) const;

            const Core::Matrix<T>& weights() const { return m_weights; }
            T bias() const { return m_bias; }
        };

    }
}