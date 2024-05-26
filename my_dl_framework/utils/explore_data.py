import os
from glob import glob

import numpy as np

from my_dl_framework.data.utils import load_any_image

image_paths = [file_name for file_name in glob(os.path.join(r'C:\data\RSNA_challenge', 'training_images', '*')) if os.path.isfile(file_name)]

all_min = []
all_max = []
all_h = []
all_w = []
for image_path in image_paths:
    image = load_any_image(image_path)
    all_min.append(np.min(image))
    all_max.append(np.max(image))
    all_h.append(image.shape[1])
    all_w.append(image.shape[0])

print('min min', np.min(all_min))
print('min max', np.min(all_max), 'max max', np.max(all_max))
print('shape', np.mean(all_w), np.mean(all_h), 'minmax', np.min(all_w), np.max(all_w), np.min(all_h), np.max(all_h))
