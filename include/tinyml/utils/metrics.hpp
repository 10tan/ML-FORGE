#pragma once
#include "tinyml/core/matrix.hpp"
#include <cmath>

namespace MLForge {
    namespace Utils {

        template <typename T>
        class Metrics {
        public:
            // Mean Squared Error: MSE = (1/n) * sum((y_true - y_pred)^2)
            static T mean_squared_error(const Core::Matrix<T>& y_true, const Core::Matrix<T>& y_pred) {
                if (y_true.size() != y_pred.size()) {
                    throw std::invalid_argument("Metrics Error: Matrix size mismatch.");
                }
                
                T mse = 0;
                size_t n = y_true.size();
                const auto& true_data = y_true.data();
                const auto& pred_data = y_pred.data();

                for (size_t i = 0; i < n; ++i) {
                    T diff = true_data[i] - pred_data[i];
                    mse += diff * diff;
                }
                
                return mse / static_cast<T>(n);
            }

            // Root Mean Squared Error: RMSE = sqrt(MSE)
            static T root_mean_squared_error(const Core::Matrix<T>& y_true, const Core::Matrix<T>& y_pred) {
                return std::sqrt(mean_squared_error(y_true, y_pred));
            }
        };

    }
}