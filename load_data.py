from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
import os

class FacesDataset(Dataset):
    """Loads AgingGAN data"""
    def __init__(self, path, transform=None):
        # getting names in directory
        self.image_names = os.listdir(path)
        self.transform = transform
    

    def load_data(self, paths: list) -> list:
        """
        Traverses across a set of path, loads data
        :param paths: set of paths to traverse
        :return: set of images
        """
        # init list of tensors
        self.tensors = []
        # initialize the transform
        transform = ToTensor()

        for path in paths:

            # initializing image
            img = Image.open(path)

            # appending tensor data
            self.tensors.append(transform(img))

        return self.tensors

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name =
        image = Image.open(image_name)




if  __name__ == "__main__":
    print(FacesDataset().load_data(['part1/1_0_0_20161219140642920.jpg']))

