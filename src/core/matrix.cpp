// src/core/matrix.cpp
// NO #includes here!

namespace MLForge {
    namespace Core {

        template <typename T> 
        Matrix<T>::Matrix() : m_rows(0), m_cols(0) {}

        template <typename T> 
        Matrix<T>::Matrix(size_t rows, size_t cols) 
            : m_rows(rows), m_cols(cols), m_data(rows * cols, T(0)) {}

        template <typename T> 
        Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<T>& flat_data) 
            : m_rows(rows), m_cols(cols), m_data(flat_data) {
            if (flat_data.size() != rows * cols) {
                throw std::invalid_argument("Matrix dimension and data size mismatch.");
            }
        }

        template <typename T> 
        T& Matrix<T>::operator()(size_t row, size_t col) { 
            return m_data[row * m_cols + col]; 
        }

        template <typename T> 
        const T& Matrix<T>::operator()(size_t row, size_t col) const { 
            return m_data[row * m_cols + col]; 
        }

        template <typename T>
        void Matrix<T>::print() const {
            for (size_t i = 0; i < m_rows; ++i) {
                std::cout << "[ ";
                for (size_t j = 0; j < m_cols; ++j) {
                    std::cout << std::setw(6) << (*this)(i, j) << " ";
                }
                std::cout << "]\n";
            }
            std::cout << std::endl;
        }

    }
}