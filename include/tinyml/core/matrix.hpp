#pragma once

#include <vector>
#include <cstddef>

namespace MLForge {
    namespace Core {

        template <typename T>
        class Matrix {
        private:
            size_t m_rows;
            size_t m_cols;
            std::vector<T> m_data;

        public:
            // Constructors
            Matrix();
            Matrix(size_t rows, size_t cols);
            Matrix(size_t rows, size_t cols, const T& initial_value);
            Matrix(size_t rows, size_t cols, const std::vector<T>& flat_data);

            // Element Access
            T& operator()(size_t row, size_t col);
            const T& operator()(size_t row, size_t col) const;

            // Dimensions
            size_t rows() const { return m_rows; }
            size_t cols() const { return m_cols; }
            size_t size() const { return m_data.size(); }
            const std::vector<T>& data() const { return m_data; }

            // Core Math Operations
            Matrix<T> transpose() const;
            Matrix<T> matmul(const Matrix<T>& other) const;
            
            // Operator Overloading for Matrix Math
            Matrix<T> operator+(const Matrix<T>& other) const; // Matrix Addition
            Matrix<T> operator-(const Matrix<T>& other) const; // Matrix Subtraction
            Matrix<T> operator*(const T& scalar) const;         // Scalar Multiplication
            
            // Utilities
            void print() const;
        };

    }
}
