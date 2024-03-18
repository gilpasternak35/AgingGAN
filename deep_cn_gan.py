# imports of relevant functions
from torch import randn, tensor, nn, flatten
from torchvision.transforms import CenterCrop
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
        self.fc_map = nn.Linear(100, 2048)
        self.first_conv_t = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride = 2, padding=0)
        self.activation = nn.LeakyReLU()

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.second_conv_t = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn2 =  nn.BatchNorm2d(num_features=32)
        self.third_conv_t = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        self.fourth_conv_t = nn.ConvTranspose2d(in_channels=16, out_channels=input_channels, kernel_size=5, stride=2, padding=0)

        # output activation
        self.output_activation = nn.Tanh()

        # initializing centercrops
        self.CCrop8, self.CCrop16, self.CCrop32, self.CCrop64 = CenterCrop(size = (8, 8)), CenterCrop(size = (16, 16)), CenterCrop(size = (32, 32)), CenterCrop(size = (64, 64))

        # initializing input shape for random noise generation
        self.input_shape = (ex_per_batch,) + input_shape

    def forward(self, device):
        """
        computes a single forward pass of the generator, returning a generated value
        :return: activation of final layer of forward pass
        """
        input = self.sample_noise().to(device)

        # resizing input
        resized_embd = self.activation(self.fc_map(input).reshape(self.input_shape[0], 128, 4, 4))

        # first deconvolution layer
        deconv1 = self.bn1(self.activation(self.first_conv_t(resized_embd)))[:,:,:8, :8]

        # second deconvolution layer
        deconv2 = self.bn2(self.activation(self.second_conv_t(deconv1)))[:,:,:16, :16]

        # third deconvolution layer
        deconv3 = self.bn3(self.activation(self.third_conv_t(deconv2)))[:,:,:32, :32]

        # computing output activation
        output = self.output_activation(self.fourth_conv_t(deconv3))[:,:,:64, :64]

        return output


    def sample_noise(self) -> tensor:
        """
        Samples noise from a random uniform distribution
        :return: a noise sample
        """
        # returns a random noise sample
        return randn(size = (self.input_shape[0], 100))


class Discriminator(nn.Module):
    """Simple MLP Discriminator Class"""
    def __init__(self, input_dims: tuple):
        """
        Initializes layers and dimensionality
        """
        super().__init__()

        # convolution, followed by a flattening and mapping to a binary output
        self.conv_layer = nn.Conv2d(in_channels = input_dims[1], out_channels = 2, kernel_size=3, padding="same")
        self.activation = nn.ReLU()
        self.linear_layer = nn.Linear(in_features= input_dims[2] * input_dims[3] * 2, out_features = 256)
        self.linear_layer2 = nn.Linear(256, out_features = 1)
        self.classification_activation = nn.Sigmoid()

    def forward(self, input: tensor) -> tensor:
        """
        Takes in an input tensor and returns scalar probability that this tensor belongs to the actual data
        :param input: the input from the generator
        :return: the probability that this tensor belongs to the actual data
        """
        # computing hidden activation - this is an image
        hidden_activation = self.activation(self.linear_layer(flatten(self.activation(self.conv_layer(input)),  start_dim=1)))

        # returning a result of linear layer applied to the flattened image.
        # Turned into probability of image being from non-generate data
        result = self.classification_activation(
                self.linear_layer2(
                    hidden_activation))

        return result