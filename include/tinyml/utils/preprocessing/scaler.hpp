#pragma once
#include "tinyml/core/matrix.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>

namespace MLForge {
    namespace Utils {
        namespace Preprocessing {

            template <typename T>
            class StandardScaler {
            private:
                std::vector<T> m_means;
                std::vector<T> m_stds;
                bool m_is_fitted = false;

            public:
                void fit(const Core::Matrix<T>& X);
                Core::Matrix<T> transform(const Core::Matrix<T>& X) const;
                Core::Matrix<T> fit_transform(const Core::Matrix<T>& X);
            };

            template <typename T>
            class MinMaxScaler {
            private:
                std::vector<T> m_mins;
                std::vector<T> m_maxs;
                bool m_is_fitted = false;

            public:
                void fit(const Core::Matrix<T>& X);
                Core::Matrix<T> transform(const Core::Matrix<T>& X) const;
                Core::Matrix<T> fit_transform(const Core::Matrix<T>& X);
            };

        }
    }
}