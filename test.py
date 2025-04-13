import scipy.io as io
import numpy as np

mat = io.loadmat('data/trainCharBound.mat')
data: np.ndarray = mat['trainCharBound']
num_samples = data.shape[1]
max_len = 0
for i in range(num_samples):
    sample = data[0, i]
    img_path = 'data/' + sample[0][0]
    label = sample[1][0]
    max_len = max(max_len, len(label))

print(max_len)