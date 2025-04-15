import torch
import torch.nn as nn
import config
from torch.optim import Adam
from utils import train_step, create_dataloader

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 batch_size: int,
                 device: str,
                 epochs: int):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
    def fit(self):
        train_dataloader = create_dataloader(data_paths=config.TRAIN_DATA_PATHS,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        train_step(model=self.model,
                   optimizer=Adam(params=self.model.parameters(),
                   lr=self.learning_rate,
                   weight_decay=1e-5),
                   train_dataloader=train_dataloader,
                   epochs=self.epochs,
                   device=self.device)
        
        