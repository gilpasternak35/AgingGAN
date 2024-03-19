# imports of relevant functions
from torch import randn, tensor, nn, flatten
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
        self.pre_conv1 = nn.Conv2d(in_channels=input_channels, out_channels = 16, stride=2, kernel_size=5,padding=2)
        self.pre_conv2 = nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=5, padding=2)
        self.pre_conv3 = nn.Conv2d(in_channels=32, out_channels=128, stride=2, kernel_size=5,
                                   padding=2)

        # to be used only for residual reshaping
        self.residual_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0)
        self.first_conv_t = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride = 2, padding=0)
        self.activation = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.second_conv_t = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.bn2 =  nn.BatchNorm2d(num_features=32)
        self.third_conv_t = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=5, stride=2, padding=0)

        # output activation
        self.output_activation = nn.Tanh()

        # initializing input shape for random noise generation
        self.input_shape = (ex_per_batch,) + input_shape

    def forward(self, device, input: torch.tensor):
        """
        computes a single forward pass of the generator, returning a generated value
        :param device: the device to perform compute on
        :param input: the conditional input to age
        :return: activation of final layer of forward pass
        """
        input = input.to(device)

        # applying prior convolutions to get to desired shape
        downsize_1 = self.activation(self.pre_conv1(input))
        downsize_2 = self.activation(self.pre_conv2(downsize_1))
        downsize_3 = self.activation(self.pre_conv3(downsize_2))

        # first deconvolution layer
        deconv1 = self.bn1(self.activation(self.first_conv_t(downsize_3)[:,:, 2:18, 2:18]))

        # second deconvolution layer
        residual_addition = self.residual_conv(downsize_1)
        deconv2 = self.bn2(self.activation(self.second_conv_t(deconv1)[:, :, 2:34, 2:34])) + residual_addition

        # third deconvolution layer
        output = self.output_activation(self.third_conv_t(deconv2)[:, :, 2:66, 2:66])

        return output


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
        self.conv_layer_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding="same")
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
        hidden_activation = self.activation(self.linear_layer(flatten(self.activation(self.conv_layer_2(self.activation(self.conv_layer(input)))),  start_dim=1)))

        # returning a result of linear layer applied to the flattened image.
        # Turned into probability of image being from non-generate data
        result = self.classification_activation(
                self.linear_layer2(
                    hidden_activation))

        return result