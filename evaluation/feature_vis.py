from deep_cn_gan import Generator
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from conditional_cn_baseline import Generator as EvalGenerator
from conditional_cn_baseline import Discriminator as EvalDiscriminator
from torchvision.transforms import Resize, Normalize, ToTensor
import torch
from PIL import Image

def visualize_features(model_path:str) -> None:
   '''
   Uses gradient based optimization to figure out which features a given filter is learning
   :param model_path: the path of the model whose features shoudl be visualized
   '''
   # selection of batch size and lr for learning process
   batch_size, lr = 16, 0.001

   # setting a device
   if torch.cuda.is_available():
        device = torch.device("cuda")
   else:
        print("GPU unavailable, using CPU instead")
        device = torch.device("cpu")


   # loading model and setting in eval mode
   model = torch.load(model_path, map_location=torch.device('cpu'))
   model.eval()

   # creating optimizing generator
   optimizing_generator = Generator(3, 100, input_shape= (16, 3, 64, 64), ex_per_batch=batch_size)

   # setting a criterion and optimizer
   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.AdamW(params=optimizing_generator.parameters(), lr = lr)

   # registering hook with desired activations
   activations = {}

   def hook_fn(module, input, output, name):
      """
      Hook function to be used to obtain desired activation
      """
      # obtaining the output for that respective layer
      activations[name] = output


   # appending hook so as to obtain the desired value from the module
   hook = model.third_conv_t.register_forward_hook(lambda self, input, output: hook_fn(self, input, output, "third_conv_t"))

   # setting an arbirary number of epochs and iterating
   num_epochs = 9000

   for epoch in range(num_epochs):
      # generating images
      generations = optimizing_generator.forward(device)

      # obtaining desired result and using them to compute loss
      temp_results = torch.zeros(16)
      desired_results = torch.fill(temp_results, 1000)

      # obtaining activation
      res = model(device, generations)

      # computing mean and loss
      mean_activation_first_filter = torch.mean(activations['third_conv_t'], dim = (1, 2,3))[:]
      loss = criterion(mean_activation_first_filter, desired_results)
      print(f"epoch: {epoch}, feature visualization loss: {loss}, mean activation: {mean_activation_first_filter}")
      loss.backward()

      # backpropagating
      optimizer.step()
      optimizer.zero_grad()


      # showing what was learned by filter
      if epoch % 1000 == 0:
         fig = plt.imshow(generations.detach()[0].permute(1,2,0))
         plt.show()

def visualize_features_disc(model_path: str):
   '''
      Uses gradient based optimization to figure out which features a discriminator is using to discriminate
      :param model_path: the path of the model whose features should be visualized
      '''
   # selection of batch size and lr for learning process
   batch_size, lr = 16, 0.001

   # setting a device
   if torch.cuda.is_available():
      device = torch.device("cuda")
   else:
      print("GPU unavailable, using CPU instead")
      device = torch.device("cpu")

   # loading model and setting in eval mode
   model = torch.load(model_path, map_location=torch.device('cpu'))
   model.eval()

   # creating optimizing generator
   optimizing_generator = EvalGenerator(3, 100, input_shape=(16, 3, 64, 64), ex_per_batch=batch_size)

   # setting a criterion and optimizer
   criterion = torch.nn.BCELoss()
   optimizer = torch.optim.AdamW(params=optimizing_generator.parameters(), lr=lr)

   # registering hook with desired activations
   activations = {}

   def hook_fn(module, input, output, name):
      """
      Hook function to be used to obtain desired activation
      """
      # obtaining the output for that respective layer
      activations[name] = output

   # appending hook so as to obtain the desired value from the module
   hook = model.linear_layer.register_forward_hook(
      lambda self, input, output: hook_fn(self, input, output, "linear_layer"))

   # setting an arbirary number of epochs and iterating
   num_epochs = 10001


   # creating an input
   gen_img = Image.open("../eval_results/img_28_before.jpg")

   # applying transforms
   resize = Resize((64, 64))
   normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   totensor = ToTensor()

   gen_input = torch.unsqueeze(resize(normalize(totensor(gen_img))), 0)

   for epoch in range(num_epochs):
      # generating images
      generations = optimizing_generator.forward(device, gen_input)

      # obtaining desired result and using them to compute loss
      temp_results = torch.zeros(1)
      desired_results = torch.fill(temp_results, 1)

      # obtaining activation
      res = model(generations)

      # computing mean and loss
      mean_activation_first_filter = res
      loss = criterion(torch.flatten(mean_activation_first_filter), desired_results)
      print(f"epoch: {epoch}, feature visualization loss: {loss}, mean activation: {mean_activation_first_filter}")
      loss.backward()

      # backpropagating
      optimizer.step()
      optimizer.zero_grad()

      # showing what was learned by filter
      if epoch % 1000 == 0:
         fig = plt.imshow(generations.detach()[0].permute(1, 2, 0))
         plt.show()
def display_saliency_map():
   pass


if __name__ == "__main__":
    visualize_features_disc( "../models/conditional_gan_disc_epoch152expconditional_baseline")


