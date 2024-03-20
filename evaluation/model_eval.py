import torch
from baseline.conditional_cn_baseline import Generator
from torchvision.transforms import Resize, Normalize, ToTensor
from PIL import Image
from load_data import FacesDataset


def eval_model(model_path: str, input_image_paths: list, save_dir: str = None, display:bool=True) -> None:
    """
    Loads a model, computes inference on a set of input image paths, displays them, and saves them in a corresponding directory
    """
    # loading generator
    gen_model = Generator(3, 0, (3, 64, 64), len(input_image_paths))

    # loading input image paths
    dataset = FacesDataset("../eval_data", "eval", 15)

