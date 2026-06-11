#include "mlforge.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "====================================================" << std::endl;
    // Calling you by your name, Tanmay, to mark this massive milestone!
    std::cout << "🚀 ML-FORGE: Master Pipeline Sandbox Activation" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    // ------------------------------------------------------------------------
    // 1. Synthesize Mock Multi-Dimensional Data (12 samples, 2 features)
    // ------------------------------------------------------------------------
    std::cout << "[Step 1] Synthesizing multi-dimensional raw feature vectors..." << std::endl;
    std::vector<float> raw_features = {
        1.2f,  2.3f,   1.5f,  2.0f,   1.1f,  2.8f,  // Group A (low values)
        2.0f,  1.9f,   1.8f,  2.5f,   2.2f,  2.1f,
        50.5f, 62.1f,  48.9f, 59.4f,  52.1f, 65.3f, // Group B (high values)
        55.2f, 60.0f,  49.5f, 58.2f,  51.0f, 63.1f
    };
    MLForge::Core::Matrix<float> X(12, 2, raw_features);

    std::vector<float> raw_targets = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    MLForge::Core::Matrix<float> y(12, 1, raw_targets);

    // ------------------------------------------------------------------------
    // 2. Partition Dataset (Train / Test Split)
    // ------------------------------------------------------------------------
    std::cout << "[Step 2] Partitioning matrix into Train and Test splits..." << std::endl;
    MLForge::Core::Matrix<float> X_train, X_test, y_train, y_test;
    MLForge::Utils::train_test_split(X, y, X_train, X_test, y_train, y_test, 0.25, 1337);
    std::cout << "  -> Training set rows: " << X_train.rows() << ", Test set rows: " << X_test.rows() << std::endl;

    // ------------------------------------------------------------------------
    // 3. Feature Engineering (Standard Scaler Optimization)
    // ------------------------------------------------------------------------
    std::cout << "[Step 3] Fitting and executing StandardScaler transformations..." << std::endl;
    MLForge::Utils::Preprocessing::StandardScaler<float> scaler;
    MLForge::Core::Matrix<float> X_train_scaled = scaler.fit_transform(X_train);
    MLForge::Core::Matrix<float> X_test_scaled = scaler.transform(X_test);

    // ------------------------------------------------------------------------
    // 4. Unsupervised Learning (K-Means Clustering Execution)
    // ------------------------------------------------------------------------
    std::cout << "[Step 4] Clustering standardized space via K-Means (K=2)..." << std::endl;
    MLForge::Unsupervised::KMeans<float> cluster_engine(2, 100);
    cluster_engine.fit(X_train_scaled, 42);
    
    std::vector<size_t> cluster_assignments = cluster_engine.predict(X_train_scaled);
    std::cout << "  -> Clustering complete. Vector states assigned successfully." << std::endl;

    // ------------------------------------------------------------------------
    // 5. Visual Analytics (Render Feature Distribution Maps)
    // ------------------------------------------------------------------------
    std::cout << "[Step 5] Launching ASCII Data Visualization Canvas Engines..." << std::endl;
    
    // Split columns to feed a 2D Scatter Plot layout mapping Col 0 vs Col 1
    MLForge::Core::Matrix<float> x_coords(X_train_scaled.rows(), 1);
    MLForge::Core::Matrix<float> y_coords(X_train_scaled.rows(), 1);
    for(size_t i = 0; i < X_train_scaled.rows(); ++i) {
        x_coords(i, 0) = X_train_scaled(i, 0);
        y_coords(i, 0) = X_train_scaled(i, 1);
    }

    MLForge::Utils::Visualization::DataVisualizer<float>::scatter_plot(x_coords, y_coords, 40, 10);
    MLForge::Utils::Visualization::DataVisualizer<float>::histogram(X_train_scaled, 0, 5, 30);

    // ------------------------------------------------------------------------
    // 6. Permanent Storage (Serialize Scaled Matrices to Disk)
    // ------------------------------------------------------------------------
    std::string export_filename = "scaled_training_data.csv";
    std::cout << "[Step 6] Exporting processed structural states to: " << export_filename << std::endl;
    X_train_scaled.to_csv(export_filename);

    std::cout << "\n====================================================" << std::endl;
    std::cout << "🎉 Pipeline Simulation Executed Successfully!" << std::endl;
    std::cout << "====================================================" << std::endl;

    return 0;
}