// src/core/matrix.cpp
// NO #includes here! Relying on unity inclusion via mlforge.hpp

namespace MLForge {
    namespace Core {

        // Constructors
        template <typename T> Matrix<T>::Matrix() : m_rows(0), m_cols(0) {}
        template <typename T> Matrix<T>::Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, T(0)) {}
        template <typename T> Matrix<T>::Matrix(size_t rows, size_t cols, const T& initial_value) : m_rows(rows), m_cols(cols), m_data(rows * cols, initial_value) {}
        template <typename T> Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<T>& flat_data) : m_rows(rows), m_cols(cols), m_data(flat_data) {
            if (flat_data.size() != rows * cols) throw std::invalid_argument("Matrix dimensions do not match data size.");
        }

        // Element Access
        template <typename T> T& Matrix<T>::operator()(size_t row, size_t col) { return m_data[row * m_cols + col]; }
        template <typename T> const T& Matrix<T>::operator()(size_t row, size_t col) const { return m_data[row * m_cols + col]; }

        // Transpose: Flips rows and columns
        template <typename T>
        Matrix<T> Matrix<T>::transpose() const {
            Matrix<T> result(m_cols, m_rows);
            for (size_t i = 0; i < m_rows; ++i) {
                for (size_t j = 0; j < m_cols; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        // Matrix Multiplication: O(N^3) standard dot product routine
        template <typename T>
        Matrix<T> Matrix<T>::matmul(const Matrix<T>& other) const {
            if (m_cols != other.rows()) throw std::invalid_argument("Matrix multiplication dimension mismatch.");
            Matrix<T> result(m_rows, other.cols(), T(0));
            for (size_t i = 0; i < m_rows; ++i) {
                for (size_t j = 0; j < other.cols(); ++j) {
                    T sum = 0;
                    for (size_t k = 0; k < m_cols; ++k) {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

        // Matrix Addition (Element-wise)
        template <typename T>
        Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
            if (m_rows != other.rows() || m_cols != other.cols()) throw std::invalid_argument("Matrix addition size mismatch.");
            Matrix<T> result(m_rows, m_cols);
            for (size_t i = 0; i < m_data.size(); ++i) {
                result.m_data[i] = this->m_data[i] + other.m_data[i];
            }
            return result;
        }

        // Matrix Subtraction (Element-wise)
        template <typename T>
        Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
            if (m_rows != other.rows() || m_cols != other.cols()) throw std::invalid_argument("Matrix subtraction size mismatch.");
            Matrix<T> result(m_rows, m_cols);
            for (size_t i = 0; i < m_data.size(); ++i) {
                result.m_data[i] = this->m_data[i] - other.m_data[i];
            }
            return result;
        }

        // Scalar Multiplication
        template <typename T>
        Matrix<T> Matrix<T>::operator*(const T& scalar) const {
            Matrix<T> result(m_rows, m_cols);
            for (size_t i = 0; i < m_data.size(); ++i) {
                result.m_data[i] = this->m_data[i] * scalar;
            }
            return result;
        }

        // Print Utility
        template <typename T>
        void Matrix<T>::print() const {
            for (size_t i = 0; i < m_rows; ++i) {
                std::cout << "[ ";
                for (size_t j = 0; j < m_cols; ++j) {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)(i, j) << " ";
                }
                std::cout << "]\n";
            }
            std::cout << std::endl;
        }

    }
}
