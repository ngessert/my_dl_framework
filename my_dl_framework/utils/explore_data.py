import numpy as np
import os
from glob import glob
from my_dl_framework.training.utils import load_any_image

image_paths = [file_name for file_name in glob(os.path.join(r"C:\data\RSNA_challenge", "training_images", "*"))
               if os.path.isfile(file_name)]


all_min = []
all_max = []
for image_path in image_paths:
    image = load_any_image(image_path)
    all_min.append(np.min(image))
    all_max.append(np.max(image))

print("min min", np.min(all_min))
print("min max", np.min(all_max), "max max", np.max(all_max))
