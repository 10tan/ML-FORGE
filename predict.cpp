#include "mlforge.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>

// 🎯 POLYMORPHIC INLINE ACTIVATION WRAPPER
template <typename T>
class InlineReLU : public MLForge::NeuralNetwork::Layer<T> {
public:
    InlineReLU() = default;
    MLForge::Core::Tensor<T> forward(const MLForge::Core::Tensor<T>& input) override {
        auto shape = input.shape();
        auto data = input.data(); 
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] < T(0)) data[i] = T(0);
        }
        return MLForge::Core::Tensor<T>(shape, data);
    }
    MLForge::Core::Tensor<T> backward(const MLForge::Core::Tensor<T>&) override { return {}; }
    void update_parameters(T) override {}
    void serialize(std::ofstream&) const override {}
    void deserialize(std::ifstream&) override {}
    std::string layer_type() const override { return "ReLU"; }
};

int main() {
    using namespace MLForge::NeuralNetwork;
    using namespace MLForge::Core;

    std::cout << "====================================================================\n";
    std::cout << " 🚀 PRODUCTION INFERENCE ENGINE: LOADING WINE_MODEL.BIN\n";
    std::cout << "====================================================================\n";

    // 1. Reconstruct the exact same architectural sequence shape layout
    std::vector<std::unique_ptr<Layer<double>>> net;
    net.push_back(std::make_unique<Dense<double>>(13, 16)); 
    net.push_back(std::make_unique<InlineReLU<double>>());  
    net.push_back(std::make_unique<Dense<double>>(16, 3));  

    // 2. Open the binary weights gateway file and read parameter blocks
    const std::string root_dir = PROJECT_ROOT_DIR;
    const std::string model_path = root_dir + "/wine_model.bin";
    std::ifstream model_file(model_path, std::ios::binary);
    
    if (!model_file.is_open()) {
        std::cerr << "❌ Execution Halt: Could not locate missing '" << model_path << "' file!\n";
        return 1;
    }

    std::cout << "📥 Deserializing weight parameters from disk into memory matrices...\n";
    for (const auto& layer : net) {
        layer->deserialize(model_file);
    }
    model_file.close();
    std::cout << "✅ Network parameters loaded successfully!\n\n";

    // 3. Provide a raw, un-normalized sample of an unseen wine profile
    // (Note: In a real app, you would pass these features through your calculated training min/max bounds!)
    // Example layout matching: [Alcohol, Malic Acid, Ash, Alkalinity, Magnesium, ..., Proline]
    std::vector<double> custom_wine_features = {
        0.75, 0.22, 0.51, 0.44, 0.39, 0.62, 0.71, 0.15, 0.48, 0.55, 0.28, 0.82, 0.64
    };

    Tensor<double> input_tensor({1, 13}, custom_wine_features);
    std::cout << "🍷 Processing input chemical vector profile...\n";

    // 4. Run a pure, lightweight Forward Pass
    auto logits = input_tensor;
    for (const auto& layer : net) {
        logits = layer->forward(logits);
    }

    // 5. Apply Softmax to extract probabilities
    SoftmaxCrossEntropy<double> loss_fn;
    // We pass a dummy target vector just to trigger the internal probabilities array matrix comfortably
    loss_fn.forward(logits, {0}); 
    auto probs = loss_fn.probabilities();

    // 6. Print class distribution confidence score metrics
    std::cout << "\n📊 Model Prediction Output Distribution:\n";
    std::cout << "---------------------------------------\n";
    std::cout << "🍇 Cultivar Class 0 Confidence: " << std::fixed << std::setprecision(2) << probs({0, 0}) * 100.0 << "%\n";
    std::cout << "🍇 Cultivar Class 1 Confidence: " << std::fixed << std::setprecision(2) << probs({0, 1}) * 100.0 << "%\n";
    std::cout << "🍇 Cultivar Class 2 Confidence: " << std::fixed << std::setprecision(2) << probs({0, 2}) * 100.0 << "%\n";
    std::cout << "---------------------------------------\n";

    // Determine highest scoring index identity
    size_t predicted_class = 0;
    double max_prob = probs({0, 0});
    for (size_t c = 1; c < 3; ++c) {
        if (probs({0, c}) > max_prob) {
            max_prob = probs({0, c});
            predicted_class = c;
        }
    }

    std::cout << "🎯 Winning Verdict: Classified as Cultivar [ " << predicted_class << " ] with " 
              << max_prob * 100.0 << "% confidence!\n";
    std::cout << "====================================================================\n";

    return 0;
}