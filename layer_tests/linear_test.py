import torch
import time
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define input size, output size, and batch size
input_size = 1000
output_size = 100
batch_size = 64

# Create a PyTorch linear layer
linear_layer = torch.nn.Linear(input_size, output_size)

# Create random input data
input_data = torch.randn(batch_size, input_size)



# Verify the layer is working correctly
def test_linear_layer():
    # Forward pass
    output = linear_layer(input_data)
    
    # Check output shape
    assert output.shape == (batch_size, output_size), f"Expected shape {(batch_size, output_size)}, but got {output.shape}"
    
    # Manual calculation for verification
    manual_output = torch.matmul(input_data, linear_layer.weight.t()) + linear_layer.bias
    
    # Check if PyTorch output matches manual calculation
    assert torch.allclose(output, manual_output, atol=1e-6), "PyTorch output doesn't match manual calculation"
    
    print("Linear layer is working correctly!")



# Measure time for forward pass
def measure_time():
    # Warm-up run
    linear_layer(input_data)
    
    # Number of iterations for timing
    num_iterations = 1000
    
    # Time the forward pass
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        linear_layer(input_data)
    end_time = time.perf_counter()
    
    # Calculate average time per forward pass
    avg_time = (end_time - start_time) / num_iterations
    print(f"Average time per forward pass: {avg_time*1e6:.2f} microseconds")

if __name__ == "__main__":
    test_linear_layer()
    measure_time()