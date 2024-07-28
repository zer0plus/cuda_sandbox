import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import glob


test_path = f"./datasets/manual_test/test_image_1_label_0.png"
test = Image.open(test_path).convert('L')  # Convert to grayscale

print("Loading MNIST dataset...")
# Load the MNIST dataset
data = np.load("/home/user/datasets/mnist.npz")
x_train, y_train = data['x_train'], data['y_train'] 
x_test, y_test = data['x_test'], data['y_test']

print("Normalizing and converting data to torch tensors...")
# Normalize and convert data to torch tensors
x_train = torch.tensor(x_train / 255.0, dtype=torch.float32)
x_test = torch.tensor(x_test / 255.0, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long) 
y_test = torch.tensor(y_test, dtype=torch.long)

print("Adding channel dimension to images...")
# Add a channel dimension to the images
x_train = x_train.unsqueeze(1)  
x_test = x_test.unsqueeze(1)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

print("Creating data loaders...")
# Create data loaders 
train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=64)
test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=64)

print("Defining CNN model...")

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Training the model...")
# Train the model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    print(f"Epoch {epoch+1} running...")
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

print("Evaluating on test set...")  
# Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct // total} %')

# Save the final trained model
torch.save(model.state_dict(), "cnn_linear_model_02(with pooling).pth")

print("Evaluating on saved test images...")
# Evaluate on the 5 saved test images

print("Evaluating on saved test images...")
# Evaluate on all the images in the folder
image_folder = "./datasets/manual_test/"
image_paths = glob.glob(image_folder + "*.png")  # Get all PNG images in the folder

for image_path in image_paths:
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)
    image = torch.from_numpy(image).float() / 255.0  # Convert to tensor and normalize
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        
    print(f"Image {image_path} - Predicted label: {predicted.item()}")