import numpy as np
import torch

np_data = np.arange(6).reshape((2, 3)) 
# print(np_data)
torch_data = torch.from_numpy(np_data)
print(torch_data)
tensor2array = torch_data.numpy()
print(tensor2array)

data = [-1, -2, 1, -2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy:', np.abs(data),
    '\ntorch:', torch.abs(tensor)
)

print(
    '\nsin',
    '\nnumpy:', np.sin(data),
    '\ntorch:', torch.sin(tensor)
)

print(
    '\nsin',
    '\nnumpy:', np.mean(data),
    '\ntorch:', torch.mean(tensor)
)

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)

print(
    '\nmatrix multiplication(matmul)',
    '\nnumpy:', np.matmul(data, data),
    '\ntorch:', torch.mm(tensor, tensor)
)

data = np.array(data)
print(
    '\nmatrix multiplication(dot)',
    '\nnumpy:', data.dot(data),
    '\ntorch:', torch.mm(tensor, tensor)
)