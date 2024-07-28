import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import os

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Create the directory to save the images
output_dir = "/home/user/datasets/manual_test"
os.makedirs(output_dir, exist_ok=True)

# Select 5 random indices from the test set
indices = np.random.choice(len(x_test), 5, replace=False)

# Save the selected images as PNG files
for i, idx in enumerate(indices):
    img = x_test[idx]
    label = y_test[idx]
    
    # Create a PIL image from the numpy array
    pil_img = Image.fromarray(img)
    
    # Save the image as PNG
    file_path = os.path.join(output_dir, f"test_image_{i+1}_label_{label}.png")
    pil_img.save(file_path, "PNG")
    
    print(f"Saved image {i+1} with label {label} at {file_path}")