#include <torch/script.h>
#include <torch/cuda.h>
#include <iostream>
#include <memory>
#include <chrono>

int main() {
    try {
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            std::cerr << "CUDA is not available" << std::endl;
            return -1;
        }

        // Load the scripted model
        torch::jit::script::Module model = torch::jit::load("../resnet50_traced_model.pt", torch::kCUDA);
        model.eval();  // Set the model to evaluation mode

        // Create a sample input tensor
        // Assuming the input shape is [1, 3, 224, 224] (batch_size, channels, height, width)
        torch::Tensor input = torch::rand({1, 3, 224, 224}).to(torch::kCUDA);

        // Warm-up run
        {
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            model.forward(inputs);
        }

        // Perform inference with timing
        torch::NoGradGuard no_grad;  // Disable gradient computation
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        auto start = std::chrono::high_resolution_clock::now();

        at::Tensor output = model.forward(inputs).toTensor();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end - start;

        // Process the output (e.g., get the predicted class)
        auto predicted_class = output.argmax(1);

        std::cout << "Predicted class: " << predicted_class.item<int64_t>() << std::endl;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}