# imports of relevant functions
from torch import rand, tensor, nn, flatten
import torch

# using gpu if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("GPU unavailable, using CPU instead")
    device = torch.device("cpu")



class Generator(nn.Module):
    """Simple Conv Generator"""
    def __init__(self, input_channels: int, hidden_channels:int, input_shape: tuple, ex_per_batch: int):
        """
        Initializes layers of an MLP Generator
        :param input_channels: the number of the channels of the input given
        :param hidden_size: the number of channels of the hidden unit
        :param input_shape: the input shape, to be used for noise generation
        :param ex_per_batch: the number of examples in a given batch
        """
        super().__init__()

        # initializing layers - unet style sequential encoder and decoder
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride = 2, padding=1),
                                     nn.GELU(),
                                     nn.BatchNorm2d(num_features=16), nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.GELU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), nn.GELU())

        # more unet style sequential encoder and decoder
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride = 2, padding=0, output_padding=0),
                                     nn.GELU(), nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1), nn.GELU(),
                                     nn.ConvTranspose2d(in_channels=16, out_channels=input_channels, kernel_size=3, stride=2, padding=1))


        # output activation
        self.output_activation = nn.Softmax()

        # initializing input shape for random noise generation
        self.input_shape = (ex_per_batch,) + input_shape

    def forward(self, device):
        """
        computes a single forward pass of the generator, returning a generated value
        :return: activation of final layer of forward pass
        """
        input = self.sample_noise().to(device)

        # applying encoder and decoder
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        # cropping to original size
        decoded_cropped = decoded[:, :, :self.input_shape[2], :self.input_shape[3]]

        # returning cropped tensor of activations
        return self.output_activation(decoded_cropped)




    def sample_noise(self) -> tensor:
        """
        Samples noise from a random uniform distribution
        :return: a noise sample
        """
        # returns a random noise sample
        return rand(self.input_shape)


class Discriminator(nn.Module):
    """Simple MLP Discriminator Class"""
    def __init__(self, input_dims: tuple):
        """
        Initializes layers and dimensionality
        """
        super().__init__()

        # convolution, followed by a flattening and mapping to a binary output
        self.conv_layer = nn.Conv2d(in_channels = input_dims[1], out_channels = 1,kernel_size=3, padding="same")
        self.activation = nn.ReLU()
        self.linear_layer = nn.Linear(in_features= input_dims[2] * input_dims[3], out_features = 1)
        self.classification_activation = nn.Sigmoid()

    def forward(self, input: tensor) -> tensor:
        """
        Takes in an input tensor and returns scalar probability that this tensor belongs to the actual data
        :param input: the input from the generator
        :return: the probability that this tensor belongs to the actual data
        """
        # computing hidden activation - this is an image
        hidden_activation = self.activation(self.conv_layer(input))

        # returning a result of linear layer applied to the flattened image.
        # Turned into probability of image being from non-generate data
        result = self.classification_activation(
                self.linear_layer(
                    flatten(hidden_activation, start_dim=1)))

        return result