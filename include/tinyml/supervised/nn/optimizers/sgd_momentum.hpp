#pragma once
#include "tinyml/supervised/nn/optimizers/optimizer.hpp"

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Optimizers {

                template <typename T>
                class SGDMomentum : public Optimizer<T> {
                private:
                    T m_lr;
                    T m_momentum;
                    Core::Matrix<T> m_vW;
                    Core::Matrix<T> m_vb;
                    bool m_initialized = false;

                public:
                    SGDMomentum(T learning_rate = T(0.01), T momentum = T(0.9))
                        : m_lr(learning_rate), m_momentum(momentum) {}

                    // 1. REMOVED template<typename T> prefix here
                    // 2. REMOVED SGDMomentum<T>:: qualification
                    void update(Core::Matrix<T>& weights, Core::Matrix<T>& bias,
                                const Core::Matrix<T>& dW, const Core::Matrix<T>& db) override {
                        if (!m_initialized) {
                            m_vW = Core::Matrix<T>(dW.rows(), dW.cols());
                            m_vb = Core::Matrix<T>(db.rows(), db.cols());
                            m_initialized = true;
                        }

                        // Update Weight matrix elements
                        for (size_t r = 0; r < dW.rows(); ++r) {
                            for (size_t c = 0; c < dW.cols(); ++c) {
                                m_vW(r, c) = m_momentum * m_vW(r, c) + m_lr * dW(r, c);
                                weights(r, c) -= m_vW(r, c);
                            }
                        }

                        // Update Bias matrix elements
                        for (size_t r = 0; r < db.rows(); ++r) {
                            for (size_t c = 0; c < db.cols(); ++c) {
                                m_vb(r, c) = m_momentum * m_vb(r, c) + m_lr * db(r, c);
                                bias(r, c) -= m_vb(r, c);
                            }
                        }
                    }
                };

            }
        }
    }
}