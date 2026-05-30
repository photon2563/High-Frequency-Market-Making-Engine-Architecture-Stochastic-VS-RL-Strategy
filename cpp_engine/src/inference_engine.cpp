#include <torch/script.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>

// Simulated AS Model for latency benchmarking
class AvellanedaStoikovCpp {
public:
    double gamma;
    double sigma;
    double k;
    
    AvellanedaStoikovCpp(double gamma = 0.1, double sigma = 0.1, double k = 1.5)
        : gamma(gamma), sigma(sigma), k(k) {}
        
    void get_quotes(double S, int q, double time_remaining, double& ask_spread, double& bid_spread) {
        // Reservation price
        double res_price = S - (q * gamma * (sigma * sigma) * time_remaining);
        
        // Optimal spread
        double optimal_spread = gamma * (sigma * sigma) * time_remaining + (2.0 / gamma) * std::log(1.0 + (gamma / k));
        
        double half_spread = optimal_spread / 2.0;
        
        double ask_price = res_price + half_spread;
        double bid_price = res_price - half_spread;
        
        ask_spread = std::max(0.01, ask_price - S);
        bid_spread = std::max(0.01, S - bid_price);
    }
};

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inference_engine <path-to-exported-a2c-actor.pt>\n";
        return -1;
    }
    
    // Load the TorchScript model
    std::cout << "[Info] Loading A2C Model...\n";
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "[Error] Error loading the model: " << e.what() << "\n";
        return -1;
    }
    std::cout << "[Info] A2C Model loaded successfully.\n\n";

    // Initialize AS baseline
    AvellanedaStoikovCpp as_model(0.1, 0.1, 1.5);
    
    // Dummy state observation tensor [Mid-price, Inventory, Time_rem, OFI, CumNotional]
    double S = 100.0;
    double q = 5.0;
    double time_rem = 0.5;
    double ofi = 0.2;
    double cum_not = 50.0;
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor({ {S, q, time_rem, ofi, cum_not} }));
    
    // Warmup the JIT compiler
    module.forward(inputs);
    
    // Measure A2C Latency
    std::cout << "--- Measuring A2C (LibTorch) Inference Latency ---\n";
    const int num_iterations = 1000;
    
    auto start_a2c = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        // forward pass
        auto output = module.forward(inputs).toTensor();
    }
    auto end_a2c = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::micro> diff_a2c = end_a2c - start_a2c;
    double avg_a2c_latency = diff_a2c.count() / num_iterations;
    
    // Test forward pass extraction
    auto a2c_output = module.forward(inputs).toTensor();
    std::cout << "A2C Output (Spread Offsets): [" 
              << a2c_output[0][0].item<float>() << ", " 
              << a2c_output[0][1].item<float>() << "]\n";
    std::cout << "Average A2C Latency: " << avg_a2c_latency << " microseconds per tick.\n\n";
    
    
    // Measure AS Latency
    std::cout << "--- Measuring Avellaneda-Stoikov C++ Latency ---\n";
    double dummy_ask, dummy_bid;
    
    auto start_as = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        as_model.get_quotes(S, static_cast<int>(q), time_rem, dummy_ask, dummy_bid);
    }
    auto end_as = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::nano> diff_as = end_as - start_as;
    double avg_as_latency = diff_as.count() / num_iterations;
    
    std::cout << "AS Output Spreads: Ask=" << dummy_ask << ", Bid=" << dummy_bid << "\n";
    std::cout << "Average AS Latency: " << avg_as_latency << " nanoseconds per tick.\n\n";
    
    std::cout << "[Result] AS is " << (avg_a2c_latency * 1000.0) / avg_as_latency << "x faster than LibTorch DRL.\n";

    return 0;
}
