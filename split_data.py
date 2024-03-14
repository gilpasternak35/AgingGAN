import os
import shutil

path_to_data = "utkcropped"

img_paths = [os.path.join(path_to_data, fname) for fname in os.listdir(path_to_data)]

out_dir = "pytorch-CycleGAN-and-pix2pix/datasets/young2old_cropped"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    os.mkdir(out_dir + "/trainA")
    os.mkdir(out_dir + "/trainB")

for img_path in img_paths:
    img_name = os.path.basename(img_path).split('/')[-1]
    age = int(img_name.split('_')[0])
    if 18 <= age <= 25:
        shutil.copy(img_path, out_dir + "/trainA/" + img_name)

    if 70 <= age:
        shutil.copy(img_path, out_dir + "/trainB/" + img_name)