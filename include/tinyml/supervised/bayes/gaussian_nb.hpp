#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace MLForge {
    namespace Bayes {

        template <typename T>
        class GaussianNB {
        private:
            // Holds the computed Gaussian parameters for each class
            struct ClassStatistics {
                T prior;
                std::vector<T> means;
                std::vector<T> variances;
            };

            std::map<size_t, ClassStatistics> m_classes;
            bool m_is_fitted = false;
            
            // Mathematical constants calculated at compile-time/construction
            const T m_pi = std::acos(-T(1.0));
            const T m_var_smoothing = T(1e-9); // Prevents division-by-zero on zero-variance features

        public:
            GaussianNB() = default;

            /**
             * @brief Fits the model by calculating the class priors, feature means, and variances.
             * @param X High-dimensional training feature matrix.
             * @param y Vector of training labels matching the rows of X.
             */
            void fit(const Core::Matrix<T>& X, const std::vector<size_t>& y) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                if (num_samples != y.size() || num_samples == 0) {
                    throw std::runtime_error("GaussianNB Error: Feature row count must match label vector dimension.");
                }

                m_classes.clear();

                // 1. Map label values to the specific sample row indices belonging to them
                std::map<size_t, std::vector<size_t>> class_map;
                for (size_t i = 0; i < num_samples; ++i) {
                    class_map[y[i]].push_back(i);
                }

                // 2. Iterate through each class to compute maximum-likelihood parameters
                for (auto const& [class_id, indices] : class_map) {
                    ClassStatistics stats;
                    T count = static_cast<T>(indices.size());
                    
                    // Prior probability: P(c) = count_c / total_count
                    stats.prior = count / static_cast<T>(num_samples);
                    stats.means.assign(num_features, T(0.0));
                    stats.variances.assign(num_features, T(0.0));

                    // Calculate the Mean (mu) for each column feature
                    for (size_t idx : indices) {
                        for (size_t j = 0; j < num_features; ++j) {
                            stats.means[j] += X(idx, j);
                        }
                    }
                    for (size_t j = 0; j < num_features; ++j) {
                        stats.means[j] /= count;
                    }

                    // Calculate the Variance (sigma^2) for each column feature
                    for (size_t idx : indices) {
                        for (size_t j = 0; j < num_features; ++j) {
                            T diff = X(idx, j) - stats.means[j];
                            stats.variances[j] += diff * diff;
                        }
                    }
                    for (size_t j = 0; j < num_features; ++j) {
                        // Apply smoothing factor to protect against numerical instability
                        stats.variances[j] = (stats.variances[j] / count) + m_var_smoothing;
                    }

                    // Store parameters into the class lookup database
                    m_classes[class_id] = stats;
                }

                m_is_fitted = true;
            }

            /**
             * @brief Predicts class labels for incoming sample rows using log-space posteriors.
             * @param X Feature matrix to evaluate.
             */
            std::vector<size_t> predict(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) {
                    throw std::runtime_error("GaussianNB Error: Model cannot predict before running fit().");
                }

                size_t num_samples = X.rows();
                size_t num_features = X.cols();
                std::vector<size_t> predictions(num_samples);

                // Process each sample row vector independently
                for (size_t i = 0; i < num_samples; ++i) {
                    T max_log_posterior = -std::numeric_limits<T>::max();
                    size_t best_class = 0;
                    bool uninitialized = true;

                    // Evaluate the likelihood of this sample belonging to each discovered class
                    for (auto const& [class_id, stats] : m_classes) {
                        
                        // Begin in log space to eliminate underflow scaling errors: log( P(c) )
                        T log_posterior = std::log(stats.prior);

                        // Accumulate the continuous independent feature likelihoods: sum( log( P(x_j | c) ) )
                        for (size_t j = 0; j < num_features; ++j) {
                            T x = X(i, j);
                            T mean = stats.means[j];
                            T var = stats.variances[j];

                            // Evaluates the standard Gaussian Probability Density Function in log space:
                            // log( 1 / sqrt(2 * pi * sigma^2) * exp( - (x - mu)^2 / (2 * sigma^2) ) )
                            T log_likelihood = -T(0.5) * std::log(T(2.0) * m_pi * var) - 
                                               ((x - mean) * (x - mean)) / (T(2.0) * var);
                            
                            log_posterior += log_likelihood;
                        }

                        // Maximize ArgMax( Log Posterior )
                        if (uninitialized || log_posterior > max_log_posterior) {
                            max_log_posterior = log_posterior;
                            best_class = class_id;
                            uninitialized = false;
                        }
                    }
                    predictions[i] = best_class;
                }

                return predictions;
            }
        };

    }
}