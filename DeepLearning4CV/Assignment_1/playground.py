import torch

# Create a tensor with a singleton dimension
a = torch.randn(1, 3, 4)

# Remove all singleton dimensions
squeezed_tensor = a.squeeze()

# Remove a specific singleton dimension (e.g., dimension 0)
squeezed_tensor_dim0 = a.squeeze(0)

print("Original tensor shape:", a.shape)
print("Squeezed tensor shape:", squeezed_tensor.shape)
print("Squeezed tensor (dim 0) shape:", squeezed_tensor_dim0.shape)