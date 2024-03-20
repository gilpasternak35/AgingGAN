import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# custom autograd function for doing guided backprop through LeakyReLU
class GuidedBackpropTanh(Function):
    @staticmethod
    def forward(ctx, input):
        # saves input tensor into ctx
        ctx.save_for_backward(input)
        # returns output of activation
        return torch.tanh(input)

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve input saved during forward pass
        input, = ctx.saved_tensors
        tanh_grad = 1 - torch.tanh(input) ** 2
        grad_input = grad_output.clone()
        grad_input *= tanh_grad
        return grad_input

class GuidedBackpropLeakyReLU(Function):
    @staticmethod
    def forward(ctx, input, negative_slope=0.01):
        ctx.save_for_backward(input)
        # sets slope of LeakyReLU
        ctx.negative_slope = negative_slope
        return F.leaky_relu(input, negative_slope=negative_slope)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        # clone gradient so as to not modify original tensor
        grad_input = grad_output.clone()
        grad_input[input < 0] *= negative_slope
        return grad_input

class GuidedBackpropModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        # Replace activation functions with guided backprop versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Tanh):
                module.register_backward_hook(self.backward_hook_tanh)
            elif isinstance(module, nn.LeakyReLU):
                module.register_backward_hook(self.backward_hook_leaky_relu)

    def backward_hook_tanh(self, module, grad_input, grad_output):
        return (GuidedBackpropTanh.apply(grad_input[0]),)

    def backward_hook_leaky_relu(self, module, grad_input, grad_output):
        return (GuidedBackpropLeakyReLU.apply(grad_input[0]),)

    def forward(self, device, input):
        print('input in forward: ', input.shape)
        return self.model(device, input)

    def __call__(self, device, input, index=None):
        print('input in __call__: ', input.shape)
        input.requires_grad = True
        output = self.forward(device, input)

        print('output shape: ', output.shape)

        if index is None:
            # Find the index of the maximum value in the flattened tensor
            index = torch.argmax(output)
            index = index.item()
        

        # Zero out all previous gradients
        self.model.zero_grad()
        print('index: ', index)

        # Backpropagate
        grad_output = torch.zeros_like(output.view(-1))  # Flatten grad_output
        print('grad output shape: ', grad_output.shape)

        # Set the corresponding index to 1 in the unflattened tensor
        grad_output[index] = 1
        grad_output = grad_output.view(output.shape)

        print('grad output shape post reshaping: ', grad_output.shape)

        output.backward(gradient=grad_output)

        gbp = input.grad
        print('gbp type: ', type(gbp))
        return gbp
