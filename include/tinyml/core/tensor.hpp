#pragma once
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <iostream>
#include <string>

namespace MLForge {
    namespace Core {

        template <typename T>
        class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            std::vector<T> m_data;
            size_t m_total_size;

            /**
             * @brief Computes memory strides to quickly index multi-dimensional coordinates.
             */
            void compute_strides() {
                m_strides.resize(m_shape.size());
                size_t current_stride = 1;
                for (size_t i = m_shape.size(); i > 0; --i) {
                    m_strides[i - 1] = current_stride;
                    current_stride *= m_shape[i - 1];
                }
            }

        public:
            // 1. Constructors
            Tensor() : m_total_size(0) {}

            Tensor(const std::vector<size_t>& shape, T initial_val = T(0.0))
                : m_shape(shape) {
                if (shape.empty()) {
                    throw std::runtime_error("Tensor Error: Shape cannot be empty.");
                }
                
                m_total_size = 1;
                for (size_t dim : m_shape) {
                    if (dim == 0) throw std::runtime_error("Tensor Error: Dimension boundaries cannot be zero.");
                    m_total_size *= dim;
                }

                m_data.assign(m_total_size, initial_val);
                compute_strides();
            }

            Tensor(const std::vector<size_t>& shape, const std::vector<T>& flat_data)
                : m_shape(shape), m_data(flat_data) {
                
                m_total_size = 1;
                for (size_t dim : m_shape) m_total_size *= dim;

                if (flat_data.size() != m_total_size) {
                    throw std::runtime_error("Tensor Error: Shape and flat data vector sizes mismatch.");
                }
                compute_strides();
            }

            // 2. Structural Dimensions
            const std::vector<size_t>& shape() const { return m_shape; }
            size_t size() const { return m_total_size; }
            const std::vector<T>& data() const { return m_data; }

            // 3. Multi-dimensional Element Access: tensor({channel, row, col})
            T& operator()(const std::vector<size_t>& indices) {
                if (indices.size() != m_shape.size()) {
                    throw std::out_of_range("Tensor Error: Coordinate index dimension layout mismatch.");
                }
                size_t flat_idx = 0;
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= m_shape[i]) {
                        throw std::out_of_range("Tensor Error: Index out of dimensional boundary spaces.");
                    }
                    flat_idx += indices[i] * m_strides[i];
                }
                return m_data[flat_idx];
            }

            const T& operator()(const std::vector<size_t>& indices) const {
                if (indices.size() != m_shape.size()) {
                    throw std::out_of_range("Tensor Error: Coordinate index dimension layout mismatch.");
                }
                size_t flat_idx = 0;
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= m_shape[i]) {
                        throw std::out_of_range("Tensor Error: Index out of dimensional boundary spaces.");
                    }
                    flat_idx += indices[i] * m_strides[i];
                }
                return m_data[flat_idx];
            }

            // 4. In-Place Element-wise Operations
            Tensor<T>& add(const Tensor<T>& other) {
                if (m_shape != other.m_shape) {
                    throw std::runtime_error("Tensor Error: Shapes must match perfectly for element-wise addition.");
                }
                for (size_t i = 0; i < m_total_size; ++i) {
                    m_data[i] += other.m_data[i];
                }
                return *this;
            }

            // 5. Shape Mutation (Reshape)
            void reshape(const std::vector<size_t>& new_shape) {
                size_t new_total = 1;
                for (size_t dim : new_shape) new_total *= dim;

                if (new_total != m_total_size) {
                    throw std::runtime_error("Tensor Error: Reshape capacity must match original spatial volumes.");
                }
                m_shape = new_shape;
                compute_strides();
            }

            // 6. Debug Print Helper
            void print(const std::string& label = "") const {
                if (!label.empty()) {
                    std::cout << "--- " << label << " (Shape: ";
                    for (size_t i = 0; i < m_shape.size(); ++i) {
                        std::cout << m_shape[i] << (i < m_shape.size() - 1 ? "x" : "");
                    }
                    std::cout << ") ---" << std::endl;
                }
                
                // For deep tensors, print flat layout with layout structure markers for simple debugging
                std::cout << "  [ ";
                for (size_t i = 0; i < m_total_size; ++i) {
                    std::cout << m_data[i] << " ";
                }
                std::cout << "]\n" << std::endl;
            }
        };

    }
}