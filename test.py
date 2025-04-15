import config
from utils import get_data_paths

data_paths, targets = get_data_paths(config.TRAIN_DATA_PATHS)
print(targets)