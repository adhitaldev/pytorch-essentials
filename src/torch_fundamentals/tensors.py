"""
Tensors, what they are and various operations on them
"""
# Tensors are fundamental data format in PyTorch that a model
# can understand. They are optimized for math ops.

import torch
import numpy as np
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

print(f"--- Running Torch Tensor Operations ---\n")
print(f"1. CREATING TENSORS")
# Creating tensors from Python list types
py_list = [1.1, 2.2, 3.4]
tensor = torch.tensor(py_list)
print(f"Created Tensor from PyList{py_list}\nTensor:{tensor}\nTensorDtype: {tensor.dtype}\n")

# From Numpy array. Note: memory is shared here. Changing one will
# change the other.
numpy_array = np.array([2.3, 4.2, 5.6])
tensor = torch.from_numpy(numpy_array)
print(f"Created Tensor from numpy arr {numpy_array}\nTensor:{tensor}\nTensorDtype: {tensor.dtype}\n")

# From pandas dataframe that is created by readig a csv
data_file = os.getenv("HOUSING_FILE_COMPLEX")
try:
    df = pd.read_csv(data_file)
    all_values = df.values
    tensor = torch.tensor(all_values)
    print(f"Created Tensor from pandas dataframe {df} from file {data_file}\nTensor:{tensor}\nTensorDtype: {tensor.dtype}\n")
except Exception as ex:
    print(f"Error creating a dataframe by reading file: {data_file}")

# Create from pre-defined values
zeros_tensor = torch.zeros(2, 3)
print(f"Predefined zero tensors {zeros_tensor}\n")
ones_tensor = torch.ones(1, 2)
print(f"Predefined one tensors {ones_tensor}")
random_tensor = torch.rand(2, 4)
print(f"Predefined random tensor {random_tensor}")

#Creating from a sequence (used to create a range of values)
range_tensor = torch.arange(0, 10, step = 1)
print(f"Predefined range tensor ({0}, {10}) {range_tensor}")

# Reshaping and manipulating tensors
print("2. RESHAPING AND MANIPULATING TENSORS")
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original Tensor {tensor} Shape: {tensor.shape}")

# Adding dimension with unsqueeze()
expanded = tensor.unsqueeze(0) # Add dimension at index 0
print(f"Adding dim After unsqueeze at idx 0 {expanded}\n Shape: {expanded.shape}")

# Removing dimension with squeeze
contracted = tensor.squeeze()
print(f"Removing dim with squeeze idx 0 {contracted}\n Shape: {contracted.shape}")

# Reshaping
reshaped = contracted.reshape(3, 2)
print(f"Reshaping last one to (3, 2): {reshaped}\nShape:{reshaped.shape}")

# Transposing
transposed = contracted.transpose(0, 1) # swaps specified dims
print(f"Transposing last one {transposed}\nShape:{transposed.shape} ")

# Combining tensors - useful for combining data from sources
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
concat_tensors = torch.cat((tensor_a, tensor_b), dim=0)
print(f"Concatenating {tensor_a} and {tensor_b} dim=0 to get {concat_tensors} of shape: {concat_tensors.shape}")

# Accessing specific parts of tensors with indexing and slicing
print("2. INDEXING AND SLICING TENSORS")
tensor = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
])
print(f"Original Tensor: {tensor}")
element_tensor = tensor[1, 2] # at row 1 and col 2
print(f"Indexing element at row 1 and col 2 : {element_tensor}")
entire_3rd_row = tensor[2]
print(f"Indexing Entire third row: {entire_3rd_row}")

# Slicing is extracting sub-tensors using [start:end:step]
first_two_rows = tensor[:2]
print(f"Slicing first two rows: {first_two_rows}")
third_col = tensor[:, 2]
print(f"Slicing third col: {third_col}")
every_other_col = tensor[:, ::2] # just define the step for the col
print(f"Every other col: {every_other_col}")

# Combining indexing and slicing
combined = tensor[:2, 2:]
print(f"First two rows, last two cols: {combined}")

# Value from a single item tensor
single_tensor = torch.tensor([4.5234])
print(f"Value from single element tensor {single_tensor}")
single_tensor = torch.tensor([4.523, 6.789])
print(f"Value from single element tensor {single_tensor}")

# Advanced Indexing - used for complex data filter with coniditions
mask = tensor > 6
mask_applied = tensor[mask]
print(f"Masked (tensor > 6) from original tensor {tensor} = {mask_applied}")

# Fancy indexing - using a tensor of indices to select specific elements
row_indices = torch.tensor([0, 2])
col_indices = torch.tensor([1, 3])
get_values = tensor[row_indices[:, None], col_indices]
print(f"After fancy indexing {tensor}")

print("2. MATHEMATICAL AND LOGICAL OPS ON TENSORS")
a = torch.tensor([1, 2, 3])
b = torch.tensor([9, 10, 11])
add = a + b
print(f"Addition of {a} and {b} = {add}")
print(f"Element wise mult = {a * b}")
print(f"Dot product using torch.matmul = {torch.matmul(a, b)}")
print(f"Dot product using '@' = {a @ b}")

# Broadcasting i.e. the automatic expansion of smaller tensor
# to match the shape of the larger tensors during arithmetic ops
b = torch.tensor([[1], [2], [3]])
c = a + b
print(f"Broadcasting {b} into {a} for addition = {c}")

# Logical operations
tensor = torch.tensor([20, 35, 19, 35, 42])
greater = tensor > 30
less = tensor <= 10
equal_to = tensor == 19
print(f"""Tensor {tensor} after logical ops.
    \nGreater than 30 {greater}
    \nLess than 10 {less} 
    \nEqual to 19 {equal_to}""")

# Element wise boolean ops
tensor_1 = torch.tensor([1, 2, 4, 3])
tensor_2 = torch.tensor([1, 2, 3, 4])
elm_and = (tensor_1 & tensor_2)
print(f"Element wise AND on {tensor_1} and {tensor_2} = {elm_and}")

# Stats
tensor = torch.tensor([1.11, 2.345, 10.12312, 110.12])
mean = tensor.mean()
std = tensor.std()
print(f"For {tensor}\nMean = {mean}\nStd = {std}")

