// src/core/tensor.cpp
// LEAVE THE TOP COMPLETELY BLANK (NO #INCLUDES)

namespace MLForge {
    namespace Core {

        template <typename T>
        void Tensor<T>::compute_strides() {
            // Keep the safety assertion outside where it belongs cleanly
            static_assert(sizeof(size_t) >= 4, "System layout bounds error");

            m_strides.resize(m_shape.size());
            size_t current_stride = 1;

            // Clean, compliant backwards traversal tracking loop
            if (!m_shape.empty()) {
                for (size_t i = m_shape.size(); i > 0; --i) {
                    m_strides[i - 1] = current_stride;
                    current_stride *= m_shape[i - 1];
                }
            }
        }

        template <typename T>
        Tensor<T>::Tensor() : m_shape({0}), m_data(0) { compute_strides(); }

        template <typename T>
        Tensor<T>::Tensor(std::initializer_list<size_t> shape) : m_shape(shape) {
            size_t total_cells = 1;
            for (auto dim : m_shape) total_cells *= dim;
            m_data.resize(total_cells, T(0));
            compute_strides();
        }

        template <typename T>
        Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& initial_value) : m_shape(shape) {
            size_t total_cells = 1;
            for (auto dim : m_shape) total_cells *= dim;
            m_data.resize(total_cells, initial_value);
            compute_strides();
        }

        // Flat mapping coordinates lookup routine
        template <typename T>
        T& Tensor<T>::operator()(std::initializer_list<size_t> indices) {
            if (indices.size() != m_shape.size()) throw std::invalid_argument("Coordinate coordinates rank mismatch.");
            size_t flat_index = 0;
            size_t idx_pos = 0;
            for (size_t index : indices) {
                if (index >= m_shape[idx_pos]) throw std::out_of_range("Tensor index bounds exceeded.");
                flat_index += index * m_strides[idx_pos];
                idx_pos++;
            }
            return m_data[flat_index];
        }

        template <typename T>
        const T& Tensor<T>::operator()(std::initializer_list<size_t> indices) const {
            if (indices.size() != m_shape.size()) throw std::invalid_argument("Coordinate coordinates rank mismatch.");
            size_t flat_index = 0;
            size_t idx_pos = 0;
            for (size_t index : indices) {
                if (index >= m_shape[idx_pos]) throw std::out_of_range("Tensor index bounds exceeded.");
                flat_index += index * m_strides[idx_pos];
                idx_pos++;
            }
            return m_data[flat_index];
        }

        template <typename T>
        Tensor<T> Tensor<T>::operator*(const T& scalar) const {
            Tensor<T> result(m_shape);
            for (size_t i = 0; i < m_data.size(); ++i) {
                result.m_data[i] = m_data[i] * scalar;
            }
            return result;
        }

        template <typename T>
        void Tensor<T>::print_summary() const {
            std::cout << "Tensor Rank: " << rank() << " | Shape: [";
            for (size_t i = 0; i < m_shape.size(); ++i) {
                std::cout << m_shape[i] << (i < m_shape.size() - 1 ? "x" : "");
            }
            std::cout << "] | Total Elements: " << size() << "\nStrides: [";
            for (size_t i = 0; i < m_strides.size(); ++i) {
                std::cout << m_strides[i] << (i < m_strides.size() - 1 ? ", " : "");
            }
            std::cout << "]\n" << std::endl;
        }

    }
}