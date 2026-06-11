#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <stdexcept>

namespace MLForge {
    namespace Bayes {

        template <typename T>
        class NaiveBayes {
        private:
            T m_alpha; // Laplace smoothing hyperparameter
            size_t m_total_samples = 0;
            bool m_is_fitted = false;

            std::map<size_t, T> m_class_priors;
            std::map<size_t, size_t> m_class_counts;
            
            // Nested lookup tracking: m_feature_counts[class_id][feature_index][feature_value] = total_occurrences
            std::map<size_t, std::map<size_t, std::map<T, size_t>>> m_feature_counts;
            
            // Track total unique vocabulary values seen per column feature across training
            std::map<size_t, std::set<T>> m_feature_vocabularies;

        public:
            // Default constructor sets alpha = 1.0 (Standard Laplace Smoothing)
            NaiveBayes(T alpha = T(1.0)) : m_alpha(alpha) {}

            /**
             * @brief Fits the discrete model by building categorical frequency histograms.
             * @param X Discretized feature occurrence count matrix.
             * @param y Associated training class labels.
             */
            void fit(const Core::Matrix<T>& X, const std::vector<size_t>& y) {
                size_t num_samples = X.rows();
                size_t num_features = X.cols();

                if (num_samples != y.size() || num_samples == 0) {
                    throw std::runtime_error("NaiveBayes Error: Input matrix row boundaries must align with label vector.");
                }

                m_class_priors.clear();
                m_class_counts.clear();
                m_feature_counts.clear();
                m_feature_vocabularies.clear();
                m_total_samples = num_samples;

                // 1. Build frequency histograms and log vocab sets
                for (size_t i = 0; i < num_samples; ++i) {
                    size_t c_id = y[i];
                    m_class_counts[c_id]++;

                    for (size_t j = 0; j < num_features; ++j) {
                        T val = X(i, j);
                        m_feature_counts[c_id][j][val]++;
                        m_feature_vocabularies[j].insert(val);
                    }
                }

                // 2. Compute baseline class priors: P(c) = N_c / N
                for (auto const& [c_id, count] : m_class_counts) {
                    m_class_priors[c_id] = static_cast<T>(count) / static_cast<T>(m_total_samples);
                }

                m_is_fitted = true;
            }

            /**
             * @brief Predicts discrete class states using Laplace-smoothed log posteriors.
             * @param X Target feature matrix to evaluate.
             */
            std::vector<size_t> predict(const Core::Matrix<T>& X) const {
                if (!m_is_fitted) {
                    throw std::runtime_error("NaiveBayes Error: Model must execute fit() before attempting predictions.");
                }

                size_t num_samples = X.rows();
                size_t num_features = X.cols();
                std::vector<size_t> predictions(num_samples);

                for (size_t i = 0; i < num_samples; ++i) {
                    T max_log_posterior = -std::numeric_limits<T>::max();
                    size_t best_class = 0;
                    bool uninitialized = true;

                    for (auto const& [c_id, prior] : m_class_priors) {
                        // Start tracking in stable log space: log( P(c) )
                        T log_posterior = std::log(prior);
                        T total_class_samples = static_cast<T>(m_class_counts.at(c_id));

                        for (size_t j = 0; j < num_features; ++j) {
                            T val = X(i, j);
                            size_t observed_count = 0;

                            // Safely crawl down nested map coordinates to extract historical match instances
                            auto class_it = m_feature_counts.find(c_id);
                            if (class_it != m_feature_counts.end()) {
                                auto feature_it = class_it->second.find(j);
                                if (feature_it != class_it->second.end()) {
                                    auto val_it = feature_it->second.find(val);
                                    if (val_it != feature_it->second.end()) {
                                        observed_count = val_it->second;
                                    }
                                }
                            }

                            // Total unique attribute values tracked for feature j across training (V)
                            size_t vocab_size = m_feature_vocabularies.count(j) ? m_feature_vocabularies.at(j).size() : 0;

                            // Execute Laplace Smoothing formulation to prevent multiplication zero collapses
                            T p_feature_given_class = (static_cast<T>(observed_count) + m_alpha) / 
                                                      (total_class_samples + m_alpha * static_cast<T>(vocab_size));
                            
                            log_posterior += std::log(p_feature_given_class);
                        }

                        // Select the argmax target class assignment
                        if (uninitialized || log_posterior > max_log_posterior) {
                            max_log_posterior = log_posterior;
                            best_class = c_id;
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