import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from torch.utils.data import DataLoader
from load_data import FacesDataset
import json

from GBP import GuidedBackpropModel  


# displays a young_old pair in the data loader
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

# generates a sample from the training dataset of your choice
def generate_sample_image(batch_idx, sample_idx):

    with open('params.json', 'r') as param_reader:
            config = json.load(param_reader)

    model_params, training_params= config['model_params'], config['training_params']

    dataset = FacesDataset(config['data_path'], mode="conditional")
    dataloader = DataLoader(dataset, batch_size = model_params['batch_size'], shuffle=False)


    for batch_num, img_label_pair in enumerate(dataloader):

        young_images, old_images = img_label_pair
        
        if batch_num == batch_idx:
            sample_young = young_images[sample_idx]
            return sample_young
        else:
            pass


def display_aged_pair(model_path, batch_idx, sample_idx):

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    sample_young = generate_sample_image(batch_idx, sample_idx)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    plt.figure(figsize=(10, 5))

    # Display the first image
    plt.subplot(1, 2, 1)
    plt.imshow(sample_young.permute(1, 2, 0))  
    plt.title('Young')
    plt.axis('off')

    sample_young = sample_young[np.newaxis,:]

    output = model(device, sample_young)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output) 
    output = np.transpose(output, (1, 2, 0))


    # Display the second image
    plt.subplot(1, 2, 2)
    plt.imshow(output)  # Use cmap='gray' for grayscale images
    plt.title('Aged')
    plt.axis('off')


def normalize(img):
    maxi = img.max()
    mini = img.min()
    return (img - mini)/(maxi-mini)


def run_gbp(model, input):

    plt.figure(figsize=(10, 5))

    permuted_inp = input.permute(1, 2, 0)
    plt.subplot(1, 2, 1)
    plt.imshow(permuted_inp)  
    plt.title('Input')
    plt.axis('off')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input = input[np.newaxis,:]

    guided_backprop = GuidedBackpropModel(model)
    guided_gradients = guided_backprop(device, input)

    guided_gradients = np.squeeze(guided_gradients)
    guided_gradients = normalize(guided_gradients)
    guided_gradients = np.transpose(guided_gradients, (1, 2, 0))


    # Display the second image
    plt.subplot(1, 2, 2)
    plt.imshow(guided_gradients)  # Use cmap='gray' for grayscale images
    plt.title('GBP')
    plt.axis('off')


def get_gbp(model, input):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input = input[np.newaxis,:]

    guided_backprop = GuidedBackpropModel(model)
    guided_gradients = guided_backprop(device, input)

    guided_gradients = np.squeeze(guided_gradients)
    guided_gradients = normalize(guided_gradients)
    guided_gradients = np.transpose(guided_gradients, (1, 2, 0))

    return guided_gradients




