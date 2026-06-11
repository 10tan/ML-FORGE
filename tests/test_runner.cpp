#include "test_matrix.hpp"
#include "test_linear_reg.hpp"
#include "test_tensor.hpp"
#include "test_metrics.hpp"
#include "test_linear_class.hpp"
#include "test_decision_tree.hpp" 
#include "test_random_forest.hpp"
#include "test_adaboost.hpp"
#include "test_gradient_boosting.hpp"
#include "test_relu_forward.hpp"
#include "test_preprocessing.hpp"
#include "test_optimization.hpp"
#include "test_kmeans.hpp"
#include "test_visualization.hpp"

#include <iostream>

int main() {
    std::cout << "🧪 Running MLForge Automated Unit Tests...\n" << std::endl;
    
    try {
        // test_matrix_addition();
        // test_linear_regression_fit();
        // test_tensor_indexing_and_scalar();
        // test_metrics_evaluation();
        // test_sigmoid_activation();
        // test_logistic_regression_classification();
        // test_decision_tree_classifier(); 
        // test_random_forest_classifier();
        // test_adaboost_classifier();
        // test_gradient_boosting_classifier();
        // test_relu_forward();
        // test_preprocessing_and_io();
        // test_optimization_and_utils();
        // test_kmeans_clustering();
        test_terminal_and_disk_plots();


        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ A test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}