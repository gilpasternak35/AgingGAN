import torch.cuda
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from load_data import FacesDataset
import matplotlib.pyplot as plt
import json
from first_gan import Generator, Discriminator

def show_images(data_path: str) -> None:
    """
    Shows images in a given batch of data
    :param data_path: the path to the data
    :return: nothing, simply show the images
    """
    # initializing dataloader upon pytorch dataset
    dset = FacesDataset(data_path)
    loader = DataLoader(dset, batch_size=4)

    # printing out by batch
    for batch in loader:
        print(len(batch))

        # showing image, re-permuting so that pixel channels appear first
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.show()


def train(config: dict) -> None:
    """
    training loop for model
    :param config: parameter configuration
    :return: Nothing, simply output results
    """

    # separating configs
    model_params, training_params= config['model_params'], config['training_params']

    # configuring dataloader
    batch_size = model_params['batch_size']
    dataset = FacesDataset(config['data_path'])
    dataloader = DataLoader(dataset, batch_size = model_params['batch_size'])


    # using gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("GPU unavailable, using CPU instead")
        device = torch.device("cpu")

    # precomputing data shape
    data_shape = (model_params['num_channels'], model_params['input_size'], model_params['input_size'])

    # obtaining number of epochs and learning rate
    num_epochs, lr = training_params['epochs'], training_params['lr']

    # initializing generator and discriminator, as well as optimizers and loss
    generator = Generator(data_shape[0], model_params['hidden_generator_channels'], data_shape, batch_size).to(device)
    discriminator = Discriminator(input_dims=(batch_size, ) + data_shape).to(device)

    # optimizers and loss
    gen_optimizer = Adam(params=generator.parameters(), lr=lr)
    disc_optimizer = Adam(params=discriminator.parameters(), lr=lr)
    criterion = BCELoss()

    # for every epoch
    for epoch in range(num_epochs):

        # loading a batch
        for batch_num, real_image_batch in enumerate(dataloader):
            # appending generated images
            generated_images = generator.forward(device).to(device)
            real_image_batch = real_image_batch.to(device)

            # computing labels
            real_labels = torch.ones((batch_size,1)).to(device)
            fake_labels = torch.zeros((batch_size,1)).to(device)

            # zeroing out gradient and getting discriminator loss
            discriminator.zero_grad()
            discriminator_outputs_real = discriminator(real_image_batch)
            discriminator_outputs_fake = discriminator(generated_images)

            # loss on real and generated
            loss_real = criterion(discriminator_outputs_real, real_labels)
            loss_fake = criterion(discriminator_outputs_fake, fake_labels)
            final_disc_loss = loss_real + loss_fake


            # back-propagating
            final_disc_loss.backward()
            disc_optimizer.step()

            for i in range(4):
                # getting generator loss - want discriminator outputs to be tricked into "real" labels
                generator.zero_grad()
                generator_labels = torch.ones(batch_size,1).to(device)
                generations = generator.forward(device)
                generator_loss = criterion(discriminator(generations), generator_labels )

                # back-propagating
                generator_loss.backward()
                gen_optimizer.step()

            # printing loss and the like
            if batch_num % 2 == 0:
                print(f"Batch num: {batch_num}, Epoch: {epoch}, Generator Loss: {generator_loss}, Discriminator Loss: {final_disc_loss}")

        # showing generated images at the end of the epoch
        plt.imshow(generator.forward(device)[0].cpu().detach().permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    # loading parameter config
    with open('params.json', 'r') as param_reader:
        config = json.load(param_reader)

    # run training loop
    train(config)