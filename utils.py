import torch
from typing import List
import string

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
    if targets.dim() == 2:
        tmp = ''
        output = torch.argmax(targets, dim=-1)
        
            

a = torch.tensor([1, 2, 3, 2,2,2,2,2,2,2, 0,0, 2, 1])
print(remove_duplicate(a))

