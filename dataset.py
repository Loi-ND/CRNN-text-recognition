from torch.utils.data import Dataset
from typing import List, Tuple
import torch
from torchvision.transforms import Normalize, Resize, Compose, ToTensor
from PIL import Image
import config

class CustomDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str], 
                 targets: List[torch.Tensor],
                 image_shape: Tuple[int, int]):
        super().__init__()
        self.image_paths = image_paths
        self.targets = targets
        mean = [0.485]
        std = [0.225]
        self.transform = Compose(
            [
                Resize((config.IMAGE_HEIGHT, config.IMAGE_HEIGHT)),
                Normalize(mean=mean, std=std),
                ToTensor()
            ]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        target = self.targets[index]

        return image, target