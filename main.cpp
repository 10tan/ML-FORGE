#include "mlforge.hpp"

int main() {
    MLForge::system_check();

    std::cout << "--- Training Linear Regression Engine ---" << std::endl;

    // Synthetic Data: 4 samples, 1 feature column (X)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    MLForge::Core::Matrix<float> X(4, 1, x_data);

    // Targets: y = 2x + 5
    std::vector<float> y_data = {7.0f, 9.0f, 11.0f, 13.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_data);

    // Initialize model: Learning Rate = 0.05, Iterations = 2000
    MLForge::Supervised::LinearRegression<float> model(0.05f, 2000);

    std::cout << "Training model..." << std::endl;
    model.fit(X, y);

    std::cout << "\nOptimized Parameters Discovered:" << std::endl;
    std::cout << "Weight (Expected ~2.0): " << model.weights()(0, 0) << std::endl;
    std::cout << "Bias   (Expected ~5.0): " << model.bias() << std::endl;

    // Predict on an unseen feature (x = 5.0 -> expected y = 15.0)
    std::vector<float> unseen_data = {5.0f};
    MLForge::Core::Matrix<float> X_new(1, 1, unseen_data);
    MLForge::Core::Matrix<float> prediction = model.predict(X_new);

    std::cout << "\nInference Test on x = 5.0:" << std::endl;
    std::cout << "Predicted y value: " << prediction(0, 0) << " (Expected: 15.0)" << std::endl;

    return 0;
}