from deep_cn_gan import Generator
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from conditional_cn_baseline import Generator as EvalGenerator
import torch

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
   hook = model.pre_conv1.register_forward_hook(lambda self, input, output: hook_fn(self, input, output, "pre_conv1"))

   # setting an arbirary number of epochs and iterating
   num_epochs = 300

   for epoch in range(num_epochs):
      # generating images
      generations = optimizing_generator.forward(device)

      # obtaining desired result and using them to compute loss
      temp_results = torch.zeros(16)
      desired_results = torch.fill(temp_results, 100000)

      # obtaining activation
      res = model(device, generations)

      # computing mean and loss
      mean_activation_first_filter = torch.mean(activations['pre_conv1'], dim = (2,3))[:,3]
      loss = criterion(mean_activation_first_filter, desired_results)
      print(f"epoch: {epoch}, feature visualization loss: {loss}, mean activation: {mean_activation_first_filter}")
      loss.backward()

      # backpropagating
      optimizer.step()
      optimizer.zero_grad()


      # showing what was learned by filter
      if epoch % 145 == 0:
         fig = plt.imshow(generations.detach()[0].permute(1,2,0))
         plt.show()


if __name__ == "__main__":
    visualize_features( "../models/conditional_gan_gen_epoch199expconditional_baseline")


