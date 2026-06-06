#pragma once

// ============================================================================
//               MLForge Master Unified Header (Umbrella)
// ============================================================================

// --- Standard Template Library Dependencies ---
#include <iostream>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <numeric>

// ============================================================================
// Global Library Configuration & Diagnostics
// ============================================================================
namespace MLForge {
    // Explicit declaration ensures src/mlforge.cpp compiles without errors
    void system_check();
}

// ============================================================================
// 1. Core Mathematical Data Structures (Templates)
// ============================================================================
#include "tinyml/core/matrix.hpp"
#include "tinyml/core/tensor.hpp"

// ============================================================================
// 2. Fundamental Architectural Interfaces
// ============================================================================
#include "tinyml/supervised/base.hpp"
#include "tinyml/unsupervised/base.hpp"

// ============================================================================
// 3. Supervised Learning Pipeline
// ============================================================================
// Linear & Regularized Regression Models
#include "tinyml/supervised/linear/linear_reg.hpp"
#include "tinyml/supervised/linear/logistic_reg.hpp"
#include "tinyml/supervised/linear/ridge.hpp"
#include "tinyml/supervised/linear/lasso.hpp"
#include "tinyml/supervised/linear/elastic_net.hpp"

// Decision Trees & Ensemble Frameworks
#include "tinyml/supervised/trees/decision_tree.hpp"
#include "tinyml/supervised/trees/random_forest.hpp"
#include "tinyml/supervised/trees/adaboost.hpp"
#include "tinyml/supervised/trees/gradient_boosting.hpp"

// Probabilistic & Bayesian Models
#include "tinyml/supervised/bayes/naive_bayes.hpp"
#include "tinyml/supervised/bayes/gaussian_nb.hpp"

// Support Vector Machines
#include "tinyml/supervised/svm/svm_classifier.hpp"
#include "tinyml/supervised/svm/kernels.hpp"

// ============================================================================
// 4. Deep Learning & Convolutions (Modular NN Engine)
// ============================================================================
#include "tinyml/supervised/nn/network.hpp"
#include "tinyml/supervised/nn/layers/layer.hpp"
#include "tinyml/supervised/nn/layers/dense.hpp"
#include "tinyml/supervised/nn/layers/conv2d.hpp"
#include "tinyml/supervised/nn/layers/pooling.hpp"
#include "tinyml/supervised/nn/layers/flatten.hpp"

// Activations & Backpropagation Optimizers
#include "tinyml/supervised/nn/activations/relu.hpp"
#include "tinyml/supervised/nn/activations/sigmoid.hpp"
#include "tinyml/supervised/nn/activations/softmax.hpp"
#include "tinyml/supervised/nn/optimizers/optimizer.hpp"
#include "tinyml/supervised/nn/optimizers/sgd.hpp"
#include "tinyml/supervised/nn/optimizers/adam.hpp"

// ============================================================================
// 5. Unsupervised Learning Pipeline
// ============================================================================
#include "tinyml/unsupervised/clustering/kmeans.hpp"
#include "tinyml/unsupervised/clustering/dbscan.hpp"
#include "tinyml/unsupervised/clustering/agglomerative.hpp"

#include "tinyml/unsupervised/dimensionality/pca.hpp"
#include "tinyml/unsupervised/dimensionality/tsne.hpp"
#include "tinyml/unsupervised/dimensionality/lda.hpp"

// ============================================================================
// 6. Common Utilities
// ============================================================================
#include "tinyml/utils/metrics.hpp"
#include "tinyml/utils/distance.hpp"
#include "tinyml/utils/data_splitter.hpp"

// ============================================================================
// 7. Core Template Implementation Linking
// Required by C++ compilation rules to resolve template instances on the fly.
// Since these are textually pasted right here, make sure their internal
// code blocks match namespace MLForge::Core!
// ============================================================================
#include "../src/core/matrix.cpp"
#include "../src/core/tensor.cpp"