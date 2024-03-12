import os
import shutil

path = "C:/Users/maxes/Downloads/part1"

img_paths = [os.path.join(path, fname) for fname in os.listdir(path)]

for img_path in img_paths:
    img_name = os.path.basename(img_path).split('/')[-1]
    age = int(img_name.split('_')[0])
    if 20 <= age <= 40:
        shutil.copy(img_path, "trainA/" + img_name)
    
    if 60 <= age <= 100:
        shutil.copy(img_path, "trainB/" + img_name)
    