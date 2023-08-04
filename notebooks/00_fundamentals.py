import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
print(torch.__version__)

# Introduction to tensors
# scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())
# vector
vector = torch.tensor([7, 7])
print(vector.ndim)
print(vector.shape)
# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX.ndim)
print(MATRIX.shape)
# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])

# Random Tensors
random_tensor = torch.rand(2, 2, 2)
print(random_tensor)
# Create a random tensore with similar shape to image tensor
random_image_tensor = torch.rand(size = (224, 224, 3))
print(random_image_tensor)

# Zeros and ones
zero_tensor = torch.zeros(size=(3,4))
print(zero_tensor)

ones_tensor = torch.ones(size=(3, 4))
print(ones_tensor.dtype)

# Create a range of tensors and tensors-like
one_to_ten = torch.arange(start=0, end=11)

# Creating tensors like
ten_zeros = torch.zeros_like(one_to_ten)
print(ten_zeros)

## Tensor data types
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # tensor data type
                               device=None, # CPU/cuda
                               requires_grad=False) # whether or not to take gradient
print(float_32_tensor)
print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

# Tensor Attributes
int_32_tensor = torch.tensor((3, 6, 9), dtype=torch.int32)
print(int_32_tensor)
print(float_32_tensor * int_32_tensor)

some_tensor = torch.rand(size=(3,4))
print(some_tensor)
print(f'Datatype of tensor: {some_tensor.dtype}')
print(f'Shape of tensor: {some_tensor.shape}')
print(f'Device of tensor: {some_tensor.device}')

# Tensor operations (addition, subtraction, multiplication, division, matrix multiplication)
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
# Try out Torch in-built function
print(torch.add(tensor, 10))
print(torch.mul(tensor, 10))

# Matrix Multiplication (dot product)
print(torch.matmul(tensor, tensor))
print(tensor @ tensor)

# Dealing with shape errors
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
# torch.matmul(tensor_A, tensor_B) -- will fail
# To fix shape issues, transpose
print(tensor_B)
print(tensor_B.T)
print(torch.matmul(tensor_A, tensor_B.T))

# Tensor aggregation
x = torch.arange(0, 100, 10)
print(torch.min(x), x.min())
# print(torch.mean(x)) -- will fail, torch.mean doesnt work for Long type
print(torch.mean(x.type(torch.float32)))
# Sum
print(torch.sum(x), x.sum()) # both using same methodology, pick one and stick with it

# Positional min/max
print(x.argmin()) # gives index value
print(x.argmax())

# Reshaping, viewing, and stacking tensors
# reshape = reshape tensor to defined shaoe
# view = return view of certain shape but keep same memory as original
# stacking = combines multiples tensors horizontally or vertically
# squeeze = removes all 1 dimensions from tensor
# unsqueeze = add a 1 dimension to a target tensor
# permute = return view with dimensions permuted

x = torch.arange(1., 11.)
print(x, x.shape)
x_reshaped = x.reshape(5, 2)
print(x_reshaped, x_reshaped.shape)
# change the view
z = x.view(5, 2)
print(z, z.shape) # changing z changes x because view shares same memory
z[ :, 0] = 5
print(z, x)
# stack tensors
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)
# squeezing
a = torch.arange(1., 10., 1)
a_reshaped = x.reshape(10, 1)
a_squeezed = a_reshaped.squeeze()
print(a_reshaped)
print(a_squeezed)
# unsqueeze
a_unsqueezed = a_squeezed.unsqueeze(dim=0)
print(a_unsqueezed.shape, a_unsqueezed)
# torch.permute returns views with dimensions permuted (re-arranged)
x_original = torch.rand(size=(224, 224, 3,))
x_permuted = x_original.permute(2, 0, 1) # these are index values
print(x_original.shape)
print(x_permuted.shape)

# Indexing
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print(x[0])
print(x[0, 0]) # same as x[0][0]
print(x[0, 0, 1])
# get all values of 0th and 1st dimensions but only index 1 of second dimension
print(x[:, :, 1])
# get all values of 0 dimension but only the 1 index value of 1st and second
print(x[:, 1, 1])
print(x[:, 2, 2])
print(x[:, :, 2])

# Pytorch tensors and numpy
# numpy array to tensor
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor) # numpy default datatype is float 64, pytorch default is float32
# change the value of the array
array = array + 1
print(array, tensor)
# Tensor to numpy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor)
print(numpy_tensor.dtype)

# Reproducability - using seeds
random_tensor_A = torch.rand(size=(3, 4))
random_tensor_B = torch.rand(size=(3, 4))
print(random_tensor_A == random_tensor_B)
# set random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(size=(3, 4))
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(size=(3, 4))
print(random_tensor_C == random_tensor_D)

# Running tensors and Pytorch objects on GPUs
# GPUs = faster computation on numbers due to CUDA + Nvidia
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# running tensors on GPU
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)
# move to gpu if available
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu.device)
# move back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu()
print(tensor_back_on_cpu.device)