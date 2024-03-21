import torch
from conditional_cn_baseline import Generator
from torchvision.transforms import Resize, Normalize, ToTensor
from PIL import Image
from torch.utils.data import DataLoader
from load_data import FacesDataset
from matplotlib import pyplot as plt
from torchvision.utils import save_image


def eval_model(model_path: str = "../models/conditional_gan_gen_epoch199expconditional_baseline", eval_data_path: str="../eval_data", save_dir: str = "../eval_results", display:bool=True, size:int = 36) -> None:
    """
    Loads a model, computes inference on a set of input image paths, displays them, and saves them in a corresponding directory
    """
    # configuring device
    device=torch.device("cpu")

    # loading generator and setting in evaluation mode
    gen_model = torch.load(model_path, map_location=torch.device("cpu"))
    gen_model.eval()

    # configuring dataloader
    dataset = FacesDataset(eval_data_path, "eval", size)
    dataloader = DataLoader(dataset, batch_size=1)

    # iterating across evaluated datapoints
    for idx, img in enumerate(dataloader):

        # generating an image
        generated = gen_model(device, img)

        # if need to display
        if display:
            # displaying original image
            plt.imshow(img[0].permute(1,2,0))
            plt.title = "Real"
            plt.show()

            # displaying new image
            plt.imshow(generated.detach()[0].permute(1, 2, 0))
            plt.title = "Generated (Aged)"
            plt.show()

        # Saving if needed
        if save_dir is not None:
            save_image(img, fp=f"{save_dir}/img_{idx}_before.jpg")
            save_image(generated, fp=f"{save_dir}/img_{idx}_generated.jpg")




if __name__ == "__main__":
    eval_model()
