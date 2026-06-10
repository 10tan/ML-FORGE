#pragma once
#include "tinyml/core/matrix.hpp"

namespace MLForge {
    namespace Unsupervised {

        template <typename T>
        class BaseTransformer {
        public:
            virtual ~BaseTransformer() = default;

            // Enforce learning internal components/clusters from raw unlabelled features
            virtual void fit(const Core::Matrix<T>& X) = 0;

            // Enforce mapping features to a lower dimension or cluster indices
            virtual Core::Matrix<T> transform(const Core::Matrix<T>& X) const = 0;

            // Convienence shortcut method to fit and map simultaneously
            virtual Core::Matrix<T> fit_transform(const Core::Matrix<T>& X) {
                fit(X);
                return transform(X);
            }
        };

    }
}