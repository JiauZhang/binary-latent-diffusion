from vq_model import VQModel
from lossers.lpips import LPIPS
from torch import nn, optim
import torch

model = VQModel()
loss_fn = LPIPS(net='vgg')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
input = torch.randn((1, 3, 32, 32), requires_grad=False)

for i in range(10):
    loss = loss_fn(input, model(input))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
