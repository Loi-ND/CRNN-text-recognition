import torch

criterion = torch.nn.CTCLoss(reduction='mean')

T = 50
C = 20
N = 16
S = 30
S_min = 10
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
loss = criterion(input, target, input_lengths, target_lengths)
print(loss.item())