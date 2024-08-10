import torch

## first part : Create tensor
x = torch.randn(3, 3, requires_grad=True)
print('Initial Tensor:\n', x)

## second part : calculate the gradient
y = x + 2
z = y * y * 3
out = z.mean()
print("\nOutput:\n", out)

out.backward()
print("\nGradients:\n", x.grad)

#禁用梯度計算
with torch.no_grad():
  y = x + 2
  print("\nTensor with no grad:\n", y)
  # y.backward()