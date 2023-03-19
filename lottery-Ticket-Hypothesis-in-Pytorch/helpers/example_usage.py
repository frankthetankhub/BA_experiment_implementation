import torch
import torchvision.models as models
import flop_counter as FLOP

inp = torch.randn(1, 3, 224, 224, device='cpu')
mod = models.resnet50()
optmizer = torch.optim.Adam(mod.parameters(), lr=0.001)

flop_counter = FLOP.FlopCounterMode(mod)
with flop_counter:
  optmizer.zero_grad()
  outputs = mod(inp) 
  loss = outputs.sum()
  loss.backward()
  optmizer.step()