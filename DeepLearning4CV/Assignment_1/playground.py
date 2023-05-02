import torch

B, N, M, P = 2, 3, 5, 4
x = torch.randn(B, N, M)
y = torch.randn(B, M, P)
z = torch.randn(B, N, P)

z[0] = x[1] @ y[1]
# print(x[1].shape, y[1].shape , (x[1] @ y[1]).shape, z[0] == x[1] @ y[1])

x = torch.tensor([[1,2],[2,3]])
y = torch.tensor([[3,4],[4,5]])
print(x @ y)
print(x.mm(y))

x = torch.rand(3,4)
y = torch.rand(3,4)
print(torch.stack((x,y), dim = 2).shape)