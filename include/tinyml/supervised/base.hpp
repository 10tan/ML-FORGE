#pragma once
#include "tinyml/core/matrix.hpp"

namespace MLForge {
    namespace Supervised {

        template <typename T>
        class BasePredictor {
        public:
            virtual ~BasePredictor() = default;

            // Enforce a unified training interface
            // X: Feature Matrix, y: Target Vector/Matrix
            virtual void fit(const Core::Matrix<T>& X, const Core::Matrix<T>& y) = 0;

            // Enforce a unified prediction interface
            // Returns the predicted outputs matching the input rows
            virtual Core::Matrix<T> predict(const Core::Matrix<T>& X) const = 0;
        };

    }
}