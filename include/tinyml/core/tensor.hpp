#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>

namespace MLForge {
    namespace Core {

        template <typename T>
        class Tensor {
        private:
            std::vector<size_t> m_shape;   // Dimensions (e.g., {batch, rows, cols})
            std::vector<size_t> m_strides; // Memory stepping layout strides
            std::vector<T> m_data;         // Continuous 1D storage allocation

            // Internal helper to compute memory strides from shape array
            void compute_strides();

        public:
            // Constructors
            Tensor();
            Tensor(std::initializer_list<size_t> shape);
            Tensor(const std::vector<size_t>& shape, const T& initial_value = T(0));

            // Element Access via N-Dimensional Coordinates (e.g., tensor({0, 1, 4}))
            T& operator()(std::initializer_list<size_t> indices);
            const T& operator()(std::initializer_list<size_t> indices) const;

            // Structural Dimensions
            const std::vector<size_t>& shape() const { return m_shape; }
            size_t rank() const { return m_shape.size(); } // Number of dimensions
            size_t size() const { return m_data.size(); }  // Total flat element count
            
            // Core Math
            Tensor<T> operator*(const T& scalar) const;    // Scalar Multiplier

            // Debug Viewer
            void print_summary() const;
        };

    }
}