#pragma once
#include "tinyml/supervised/nn/optimizers/optimizer.hpp"
#include <cmath>
#include <vector>

namespace MLForge {
    namespace Supervised {
        namespace NN {
            namespace Optimizers {

                template <typename T>
                class Adam : public Optimizer<T> {
                private:
                    T m_lr;       // Learning Rate (alpha)
                    T m_beta1;    // Exponential decay rate for the first moment estimates
                    T m_beta2;    // Exponential decay rate for the second-moment estimates
                    T m_eps;      // Tiny term for numerical stability (prevents division by zero)
                    size_t m_t;   // Timestep counter tracking updates

                    // Running state moment containers matching parameter dimensions
                    Core::Matrix<T> m_mW; // First moment vector (mean) for weights
                    Core::Matrix<T> m_vW; // Second moment vector (uncentered variance) for weights
                    Core::Matrix<T> m_mb; // First moment vector for biases
                    Core::Matrix<T> m_vb; // Second moment vector for biases
                    
                    bool m_initialized; // Toggle tracking allocation states

                public:
                    /**
                     * @brief Construct a new Adam Optimizer object using standard defaults
                     * @param learning_rate Default standard optimization step step size (0.001)
                     * @param beta1 Gradient running average decay setting (0.9)
                     * @param beta2 Squared gradient running average decay setting (0.999)
                     * @param epsilon Denominator safety parameter (1e-8)
                     */
                    Adam(T learning_rate = T(0.001), T beta1 = T(0.9), T beta2 = T(0.999), T epsilon = T(1e-8))
                        : m_lr(learning_rate), m_beta1(beta1), m_beta2(beta2), m_eps(epsilon), 
                          m_t(0), m_initialized(false) {}

                    /**
                     * @brief Performs an adaptive step parameter mutation pass based on evaluated derivatives
                     */
                    void update(Core::Matrix<T>& weights, Core::Matrix<T>& bias,
                                const Core::Matrix<T>& dW, const Core::Matrix<T>& db) override {
                        
                        // 1. Lazy allocation initialization matching exact gradient runtime sizing
                        if (!m_initialized) {
                            m_mW = Core::Matrix<T>(dW.rows(), dW.cols());
                            m_vW = Core::Matrix<T>(dW.rows(), dW.cols());
                            m_mb = Core::Matrix<T>(db.rows(), db.cols());
                            m_vb = Core::Matrix<T>(db.rows(), db.cols());
                            m_initialized = true;
                        }

                        // 2. Increment update iteration time step counter
                        m_t++;

                        // 3. Compute dynamic bias correction scalars for this iteration step
                        T bias_correction1 = T(1) - std::pow(m_beta1, m_t);
                        T bias_correction2 = T(1) - std::pow(m_beta2, m_t);

                        // ============================================================================
                        // Weight Array Parameter Processing Loop
                        // ============================================================================
                        for (size_t r = 0; r < dW.rows(); ++r) {
                            for (size_t c = 0; c < dW.cols(); ++c) {
                                // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
                                m_mW(r, c) = m_beta1 * m_mW(r, c) + (T(1) - m_beta1) * dW(r, c);
                                
                                // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
                                m_vW(r, c) = m_beta2 * m_vW(r, c) + (T(1) - m_beta2) * dW(r, c) * dW(r, c);
                                
                                // Compute bias-corrected first moment estimate
                                T m_hat = m_mW(r, c) / bias_correction1;
                                
                                // Compute bias-corrected second raw moment estimate
                                T v_hat = m_vW(r, c) / bias_correction2;
                                
                                // Mutate parameter coordinate element safely
                                weights(r, c) -= m_lr * m_hat / (std::sqrt(v_hat) + m_eps);
                            }
                        }

                        // ============================================================================
                        // Bias Array Parameter Processing Loop
                        // ============================================================================
                        for (size_t r = 0; r < db.rows(); ++r) {
                            for (size_t c = 0; c < db.cols(); ++c) {
                                // Update biased first moment estimate for bias terms
                                m_mb(r, c) = m_beta1 * m_mb(r, c) + (T(1) - m_beta1) * db(r, c);
                                
                                // Update biased second raw moment estimate for bias terms
                                m_vb(r, c) = m_beta2 * m_vb(r, c) + (T(1) - m_beta2) * db(r, c) * db(r, c);
                                
                                // Bias corrections
                                T m_hat = m_mb(r, c) / bias_correction1;
                                T v_hat = m_vb(r, c) / bias_correction2;
                                
                                // Mutate bias parameter element
                                bias(r, c) -= m_lr * m_hat / (std::sqrt(v_hat) + m_eps);
                            }
                        }
                    }
                    
                    // Utility method to wipe history if reuse across different models occurs
                    void reset() {
                        m_t = 0;
                        m_initialized = false;
                    }
                };

            }
        }
    }
}