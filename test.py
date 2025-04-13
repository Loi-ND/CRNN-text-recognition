import torch
a = torch.tensor([1, 2, 3])
samples = [a, a]

print(torch.cat(samples))