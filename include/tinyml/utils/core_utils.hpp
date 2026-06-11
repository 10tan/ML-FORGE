#pragma once
#include "tinyml/core/matrix.hpp"
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace MLForge {
    namespace Utils {

        template <typename T>
        void train_test_split(const Core::Matrix<T>& X, const Core::Matrix<T>& y,
                              Core::Matrix<T>& X_train, Core::Matrix<T>& X_test,
                              Core::Matrix<T>& y_train, Core::Matrix<T>& y_test,
                              double test_size = 0.25, unsigned int seed = 42) {
            
            if (X.rows() != y.rows()) {
                throw std::runtime_error("Split Error: X and y must have the exact same number of rows.");
            }

            size_t total_rows = X.rows();
            size_t test_rows = static_cast<size_t>(static_cast<double>(total_rows) * test_size);
            size_t train_rows = total_rows - test_rows;

            // Generate index vector and shuffle using a seeded engine
            std::vector<size_t> indices(total_rows);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 rng(seed);
            std::shuffle(indices.begin(), indices.end(), rng);

            // Resize destination matrices
            X_train = Core::Matrix<T>(train_rows, X.cols());
            X_test = Core::Matrix<T>(test_rows, X.cols());
            y_train = Core::Matrix<T>(train_rows, y.cols());
            y_test = Core::Matrix<T>(test_rows, y.cols());

            // Allocate shuffled blocks
            for (size_t i = 0; i < train_rows; ++i) {
                y_train(i, 0) = y(indices[i], 0);
                for (size_t j = 0; j < X.cols(); ++j) X_train(i, j) = X(indices[i], j);
            }
            for (size_t i = 0; i < test_rows; ++i) {
                y_test(i, 0) = y(indices[train_rows + i], 0);
                for (size_t j = 0; j < X.cols(); ++j) X_test(i, j) = X(indices[train_rows + i], j);
            }
        }

    }
}