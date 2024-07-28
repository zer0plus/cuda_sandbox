import torch
import os
import numpy as np
import time
from torch import tensor
from torch.utils.cpp_extension import load_inline
from wurlitzer import sys_pipes
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import glob

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

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



if __name__ == "__main__":
    with sys_pipes():
        print("start")