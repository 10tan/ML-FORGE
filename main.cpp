#include "mlforge.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

// 🎯 POLYMORPHIC INLINE ACTIVATION TO BYPASS CLASS INHERITANCE BLOCKS
template <typename T>
class InlineReLU : public MLForge::NeuralNetwork::Layer<T> {
private:
    MLForge::Core::Tensor<T> m_input_cache;

public:
    InlineReLU() = default;

    // 🎯 FIX: Implement the pure virtual serialization interfaces safely as empty blocks
    void serialize(std::ofstream& os) const override {
        // Activations have no parameters/weights to save to disk!
    }

    void deserialize(std::ifstream& is) override {
        // Activations have no parameters/weights to load from disk!
    }

    MLForge::Core::Tensor<T> forward(const MLForge::Core::Tensor<T>& input) override {
        m_input_cache = input;
        auto shape = input.shape();
        auto data = input.data(); 
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] < T(0)) {
                data[i] = T(0);
            }
        }
        return MLForge::Core::Tensor<T>(shape, data);
    }

    MLForge::Core::Tensor<T> backward(const MLForge::Core::Tensor<T>& output_gradient) override {
        auto shape = output_gradient.shape();
        auto grad_data = output_gradient.data();
        const auto& input_data = m_input_cache.data();

        for (size_t i = 0; i < grad_data.size(); ++i) {
            if (input_data[i] <= T(0)) {
                grad_data[i] = T(0);
            }
        }
        return MLForge::Core::Tensor<T>(shape, grad_data);
    }

    void update_parameters(T) override {
        // Activations hold no trainable weight metrics
    }

    std::string layer_type() const override { return "ReLU (Polymorphic)"; }
};

// 🍇 COGNITIVE PARSER: EXTRACTS CHEMICAL FEATURES FROM THE WINE DATASET
void load_wine_split(const std::string& path, 
                     MLForge::Core::Tensor<double>& X_tensor, 
                     std::vector<size_t>& y_vector, 
                     size_t start_row, size_t num_rows) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open Wine dataset asset at: " + path);
    }

    std::string line;
    std::getline(file, line); // Skip the csv header line

    size_t skipped = 0;
    while (skipped < start_row && std::getline(file, line)) {
        skipped++;
    }

    std::vector<double> flat_features;
    y_vector.clear();
    size_t rows_parsed = 0;

    while (std::getline(file, line) && rows_parsed < num_rows) {
        std::stringstream ss(line);
        std::string value;
        
        size_t class_label = 0;
        std::vector<double> features;

        size_t col_idx = 0;
        while (std::getline(ss, value, ',')) {
            if (col_idx == 0) {
                class_label = std::stoul(value) - 1;
            } else {
                features.push_back(std::stod(value));
            }
            col_idx++;
        }

        if (features.size() == 13) {
            flat_features.insert(flat_features.end(), features.begin(), features.end());
            y_vector.push_back(class_label);
            rows_parsed++;
        }
    }

    X_tensor = MLForge::Core::Tensor<double>({rows_parsed, 13}, flat_features);
}

// 🎯 MATRICES BOUNDARY NORMALIZER
void apply_bound_normalization(MLForge::Core::Tensor<double>& X, 
                               const std::vector<double>& min_vals, 
                               const std::vector<double>& max_vals) {
    auto shape = X.shape();
    size_t rows = shape[0];
    size_t cols = shape[1];

    for (size_t c = 0; c < cols; ++c) {
        double range = max_vals[c] - min_vals[c];
        if (range == 0.0) range = 1.0;

        for (size_t r = 0; r < rows; ++r) {
            X({r, c}) = (X({r, c}) - min_vals[c]) / range;
        }
    }
}

double calculate_accuracy(const MLForge::Core::Tensor<double>& probabilities, const std::vector<size_t>& targets) {
    size_t correct = 0;
    size_t batch_size = probabilities.shape()[0];
    size_t num_classes = probabilities.shape()[1];

    for (size_t b = 0; b < batch_size; ++b) {
        size_t highest_prob_class = 0;
        double max_prob = probabilities({b, 0});
        for (size_t c = 1; c < num_classes; ++c) {
            if (probabilities({b, c}) > max_prob) {
                max_prob = probabilities({b, c});
                highest_prob_class = c;
            }
        }
        if (highest_prob_class == targets[b]) correct++;
    }
    return (static_cast<double>(correct) / batch_size) * 100.0;
}

void print_banner(const std::string& section_name) {
    std::cout << "\n====================================================================\n";
    std::cout << " 🍇 " << section_name << "\n";
    std::cout << "====================================================================\n";
}

int main() {
    using namespace MLForge::NeuralNetwork;
    using namespace MLForge::Core;

    std::cout << "====================================================================\n";
    std::cout << " 🧪 CHEMICAL PROFILE ENGINE: DEPLOYING ON UCI WINE BENCHMARK\n";
    std::cout << "====================================================================\n";

    int user_epochs = 0;
    std::cout << "🔢 Enter the number of training epochs you want to run: ";
    if (!(std::cin >> user_epochs) || user_epochs <= 0) {
        std::cerr << "❌ Invalid entry. Defaulting to 40.\n";
        user_epochs = 40;
    }

    const std::string root_dir = PROJECT_ROOT_DIR;
    const std::string dataset_path = root_dir + "/data/uci-wine/wine.csv";

    Tensor<double> train_X, test_X;
    std::vector<size_t> train_y, test_y;

    try {
        load_wine_split(dataset_path, train_X, train_y, 0, 140);
        load_wine_split(dataset_path, test_X, test_y, 140, 38);

        auto train_shape = train_X.shape();
        size_t cols = train_shape[1];
        std::vector<double> train_min(cols), train_max(cols);

        for (size_t c = 0; c < cols; ++c) {
            train_min[c] = train_X({0, c});
            train_max[c] = train_X({0, c});
            for (size_t r = 1; r < train_shape[0]; ++r) {
                train_min[c] = std::min(train_min[c], train_X({r, c}));
                train_max[c] = std::max(train_max[c], train_X({r, c}));
            }
        }

        apply_bound_normalization(train_X, train_min, train_max);
        apply_bound_normalization(test_X, train_min, train_max);

        std::cout << "\n📈 Training Features Loaded : [" << train_X.shape()[0] << ", " << train_X.shape()[1] << "]\n";
        std::cout << "🧪 Testing Features Loaded  : [" << test_X.shape()[0] << ", " << test_X.shape()[1] << "]\n";
    } catch (const std::exception& e) {
        std::cerr << "Pipeline Halt: " << e.what() << "\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    // LAYER CONFIGURATIONS
    // ------------------------------------------------------------------------
    std::vector<std::unique_ptr<Layer<double>>> net;
    net.push_back(std::make_unique<Dense<double>>(13, 16)); 
    net.push_back(std::make_unique<InlineReLU<double>>());  
    net.push_back(std::make_unique<Dense<double>>(16, 3));  
    
    SoftmaxCrossEntropy<double> loss_fn;

    // 🛠️ GRADIENT-DRIVEN WEIGHT INITIALIZATION SHAKEUP
    // Since direct array data manipulation is read-only, we pass a random normal distribution gradient 
    // backward through the framework to break structural symmetry before the training loop starts.
    {
        std::mt19937 gen(1337);
        std::normal_distribution<double> dist(0.0, 0.5);
        
        // Match the final logit output shape format configuration [Batch Size, Classes] -> [140, 3]
        std::vector<double> random_grad_data(140 * 3);
        for (double& val : random_grad_data) val = dist(gen);
        
        Tensor<double> init_grad({140, 3}, random_grad_data);
        
        // Prime internal activation states via a dummy forward pass
        auto prime_logits = train_X;
        for (const auto& layer : net) prime_logits = layer->forward(prime_logits);
        
        // Push the random noise gradient matrix backward through the framework layers
        for (auto it = net.rbegin(); it != net.rend(); ++it) {
            init_grad = (*it)->backward(init_grad);
        }
        
        // Call an aggressive parameter update step to establish unique initial weights
        for (const auto& layer : net) {
            layer->update_parameters(0.2); 
        }
    }

    print_banner("RUNNING MULTI-CLASS GRADIENT PROPAGATION");
    std::cout << " Epoch |       Train Loss |     Train Acc (%) |   Learning Rate\n";
    std::cout << "---------------------------------------------------------------------\n";

    double lr = 0.40; 

    for (int epoch = 1; epoch <= user_epochs; ++epoch) {
        auto logits = train_X;
        for (const auto& layer : net) {
            logits = layer->forward(logits);
        }

        double loss = loss_fn.forward(logits, train_y);
        double train_acc = calculate_accuracy(loss_fn.probabilities(), train_y);

        auto grad = loss_fn.backward(train_y);
        for (auto it = net.rbegin(); it != net.rend(); ++it) {
            grad = (*it)->backward(grad);
        }

        for (const auto& layer : net) {
            layer->update_parameters(lr);
        }

        lr *= 0.999; 

        std::cout << " " << std::setw(5) << epoch 
                  << " | " << std::setw(16) << std::fixed << std::setprecision(6) << loss 
                  << " | " << std::setw(17) << std::fixed << std::setprecision(2) << train_acc 
                  << " | " << std::setw(15) << std::fixed << std::setprecision(5) << lr << "\n";
    }

    std::cout << "\n💾 Freezing network parameters and saving model to disk...\n";

    std::string save_path = root_dir + "/wine_model.bin";
    std::ofstream model_file(save_path, std::ios::binary);
    if (model_file.is_open()) {
        for (const auto& layer : net) {
            layer->serialize(model_file);
        }
        model_file.close();
        std::cout << "🌟 Model successfully saved to " << save_path << "\n";
    } else {
        std::cerr << "❌ Error: Could not open file to save weights.\n";
    }

    // ------------------------------------------------------------------------
    // GENERALIZATION TEST PASS
    // ------------------------------------------------------------------------
    print_banner("EVALUATING UNSEEN TEST SAMPLES");
    
    auto eval_logits = test_X;
    for (const auto& layer : net) {
        eval_logits = layer->forward(eval_logits);
    }
    
    double test_loss = loss_fn.forward(eval_logits, test_y);
    double test_acc = calculate_accuracy(loss_fn.probabilities(), test_y);

    std::cout << "🍷 Test Dataset Cross-Entropy Loss : " << std::fixed << std::setprecision(6) << test_loss << "\n";
    std::cout << "🎯 Test Generalization Accuracy     : " << std::fixed << std::setprecision(2) << test_acc << "%\n";
    std::cout << "====================================================================\n";

    return 0;
}