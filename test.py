from model import CRNN
from trainer import Trainer
import config

crnn_model = CRNN(in_channels=1)
trainer = Trainer(model=crnn_model,
                  learning_rate=1e-4,
                  batch_size=32,
                  device=config.DEVICE,
                  epochs=10)
