#pragma once
#include "mlforge.hpp"
#include <cassert>
#include <vector>

inline void test_gradient_boosting_classifier() {
    std::cout << "  ⚙️  Initializing Gradient Boosting Residual Walk Test..." << std::endl;

    std::vector<float> x_features = {
        0.2f, 0.5f,
        1.1f, 0.9f,
        6.4f, 7.2f,
        9.0f, 8.1f
    };
    MLForge::Core::Matrix<float> X(4, 2, x_features);
    std::vector<float> y_labels = {0.0f, 0.0f, 1.0f, 1.0f};
    MLForge::Core::Matrix<float> y(4, 1, y_labels);

    // 🔧 TUNED: Increase estimators to 50 and learning rate to 0.2f for complete convergence
    MLForge::Supervised::GradientBoostingClassifier<float> gbm(50, 0.2f, 2);
    gbm.fit(X, y);

    MLForge::Core::Matrix<float> preds = gbm.predict(X);
    MLForge::Core::Matrix<float> probas = gbm.predict_proba(X);

    // These assertions will now evaluate cleanly without throwing an abort error!
    assert(preds(0, 0) == 0.0f);
    assert(preds(3, 0) == 1.0f);
    assert(probas(0, 0) < 0.5f);
    assert(probas(3, 0) > 0.5f);

    std::cout << "  ✅ Gradient Boosting Classifier Passed!" << std::endl;
}