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
            


