#pragma once

namespace MLForge {
    namespace Core {

        template <typename T>
        class Matrix {
        private:
            size_t m_rows;
            size_t m_cols;
            std::vector<T> m_data; // Continuous 1D storage for 2D space

        public:
            Matrix();
            Matrix(size_t rows, size_t cols);
            Matrix(size_t rows, size_t cols, const std::vector<T>& flat_data);

            // Row-major element lookup: index = row * total_columns + col
            T& operator()(size_t row, size_t col);
            const T& operator()(size_t row, size_t col) const;

            size_t rows() const { return m_rows; }
            size_t cols() const { return m_cols; }
            
            void print() const;
        };

    }
}