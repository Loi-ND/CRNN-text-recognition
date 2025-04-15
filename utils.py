import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
import string
from tqdm import tqdm

chars = string.ascii_letters + string.digits
directory = {char: i+1 for i, char in enumerate(chars)}

def remove_duplicate(target: torch.Tensor) -> List[int]:
    output1 = []
    prev = -1  
    for t in target:
        t = t.item()
        if t != prev:
            output1.append(t)
            prev = t    
    output2 = []
    for x in output1:
        if x != 0:
            output2.append(x) 
    return output2
    
def encode_targets(targets: List[str]) -> List[torch.Tensor]:
    output = []
    for target in targets:
        tmp = []
        for char in target:
            tmp.append(directory[char])
        tmp = torch.tensor(tmp)
        output.append(tmp)
    return output


def decode_targets(targets: torch.Tensor):
    targets = torch.argmax(targets, dim=-1)
    if targets.dim() == 1:
        tmp = ''
        output = remove_duplicate(target=targets)
        for idx in output:
            tmp += chars[idx-1]
        return tmp
    else:
        out = []
        for target in targets:
            tmp = ''
            output = remove_duplicate(target=target)
            for idx in output:
                tmp += chars[idx-1]
            out.append(tmp)
        return out

def get_data_paths(paths: str) -> Tuple[List[str], List[torch.Tensor]]:
    image_paths = []
    targets = []
    with open(paths, mode='r') as file:
        for line in file.readlines():
            image_path, target = line.split()
            targets.append(target)
            image_paths.append(image_path)
    targets = encode_targets(targets)

    return image_paths, targets

def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               train_dataloader: DataLoader,
               epochs: int,
               device: str):
    model.to(device=device)
    for epoch in tqdm(epochs):
        num_samples = 0
        cumsum_loss = 0
        for i, (data, label) in enumerate(train_dataloader):
            num_samples += len(data)
            data = data.to(device)
            optimizer.zero_grad()

            output, loss = model(data, label)   
            cumsum_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epochs {epoch + 1} Training loss = {(cumsum_loss/num_samples):.2f}')

def evaluate(model: nn.Module,
             test_dataloader: DataLoader) -> float:
    outputs = decode_targets(outputs)
        