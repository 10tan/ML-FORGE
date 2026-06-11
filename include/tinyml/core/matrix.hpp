#pragma once
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <iostream>
#include <fstream>
namespace MLForge {
    namespace Core {

        template <typename T>
        class Matrix {
        private:
            size_t m_rows;
            size_t m_cols;
            std::vector<T> m_data;

        public:
            // 1. Constructors
            Matrix() : m_rows(0), m_cols(0) {}
            
            Matrix(size_t rows, size_t cols, T initial_val = T(0.0))
                : m_rows(rows), m_cols(cols), m_data(rows * cols, initial_val) {}

            Matrix(size_t rows, size_t cols, const std::vector<T>& flat_data)
                : m_rows(rows), m_cols(cols), m_data(flat_data) {
                if (flat_data.size() != rows * cols) {
                    throw std::runtime_error("Matrix Error: Initializer data vector size mismatch.");
                }
            }

            // 2. Structural Dimensions
            size_t rows() const { return m_rows; }
            size_t cols() const { return m_cols; }
            const std::vector<T>& data() const { return m_data; }

            // 3. Element Access Operators
            T& operator()(size_t row, size_t col) {
                if (row >= m_rows || col >= m_cols) {
                    throw std::out_of_range("Matrix Error: Index out of boundaries.");
                }
                return m_data[row * m_cols + col];
            }

            const T& operator()(size_t row, size_t col) const {
                if (row >= m_rows || col >= m_cols) {
                    throw std::out_of_range("Matrix Error: Index out of boundaries.");
                }
                return m_data[row * m_cols + col];
            }

            void to_csv(const std::string& filename) const {
                std::ofstream file(filename);
                
                if (!file.is_open()) {
                    throw std::runtime_error("Matrix I/O Error: Unable to open or create file: " + filename);
                }

                for (size_t i = 0; i < m_rows; ++i) {
                    for (size_t j = 0; j < m_cols; ++j) {
                        file << (*this)(i, j);
                        
                        // Append a comma separating columns, unless it's the last value in the row
                        if (j < m_cols - 1) {
                            file << ",";
                        }
                    }
                    file << "\n"; // Break for the next record row
                }
                
                file.close();
            }
            
            // 4. Matrix Multiplication: Matrix * Matrix
            Matrix<T> matmul(const Matrix<T>& other) const {
                if (m_cols != other.rows()) {
                    throw std::runtime_error("Matrix Error: Inner dimensions must align for multiplication.");
                }

                Matrix<T> result(m_rows, other.cols(), T(0.0));
                for (size_t i = 0; i < m_rows; ++i) {
                    for (size_t j = 0; j < other.cols(); ++j) {
                        T sum = T(0.0);
                        for (size_t k = 0; k < m_cols; ++k) {
                            sum += (*this)(i, k) * other(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
                return result;
            }

            // 5. Matrix Transpose Engine
            Matrix<T> transpose() const {
                Matrix<T> result(m_cols, m_rows);
                for (size_t i = 0; i < m_rows; ++i) {
                    for (size_t j = 0; j < m_cols; ++j) {
                        result(j, i) = (*this)(i, j);
                    }
                }
                return result;
            }

            // 6. Print Debug Helper
            void print(const std::string& label = "") const {
                if (!label.empty()) {
                    std::cout << "--- " << label << " (" << m_rows << "x" << m_cols << ") ---" << std::endl;
                }
                for (size_t i = 0; i < m_rows; ++i) {
                    std::cout << "  [ ";
                    for (size_t j = 0; j < m_cols; ++j) {
                        std::cout << (*this)(i, j) << " ";
                    }
                    std::cout << "]" << std::endl;
                }
                std::cout << std::endl;
            }
        };

    }
}