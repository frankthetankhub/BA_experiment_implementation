import torch
import numpy as np
import random

norm_rand = []
norm_rand1 = list()
torch_rand= []
torch_rand1 =[]
np_rand = []
np_rand1 = []
# for i in range(10):
#     norm_rand.append(random.randrange(0,100))
#     torch_rand.append(torch.rand(5))
#     np_rand.append(np.random.rand(10))
# print(norm_rand)
# print(torch_rand)
# print(np_rand)
random.seed(10)
#gen_torch = torch.random.manual_seed(10)
torch.manual_seed(10)
#gen= np.random.default_rng(10)
np.random.seed(10)
for i in range(1):
    norm_rand1.append(random.randrange(0,100))
    torch_rand1.append(torch.rand(5)) # ,generator=gen_torch)
    np_rand1.append(np.random.standard_normal(10)) #gen.standard_normal(10)
print(norm_rand1, torch_rand1, np_rand1)