import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from torch.utils.data import DataLoader
from load_data import FacesDataset
import json

def display_young_old_(batch_idx, sample_idx):

    with open('params.json', 'r') as param_reader:
            config = json.load(param_reader)

    model_params, training_params= config['model_params'], config['training_params']

    dataset = FacesDataset(config['data_path'], mode="conditional")
    dataloader = DataLoader(dataset, batch_size = model_params['batch_size'], shuffle=False)

    for batch_num, img_label_pair in enumerate(dataloader):

        young_images, old_images = img_label_pair

        if batch_num == batch_idx:
            print(f'There are {len(young_images)} images in the batch')
            plt.imshow(young_images[sample_idx].permute(1, 2, 0))
            plt.show()
            plt.imshow(old_images[sample_idx].permute(1, 2, 0))
            plt.show()
            break
        else:
            pass
