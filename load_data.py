from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize
import os

class FacesDataset(Dataset):
    """Loads AgingGAN data"""
    def __init__(self, path, transform=None):
        """
        Initializes PyTorch Dataset
        :param path: path to files
        :param transform: set of transforms to apply
        """
        # getting image names in directory
        self.image_names = [os.path.join(path, fname) for fname in os.listdir(path) if "jpg" in fname][:2000]

        # initializing transforms, resize, random crops
        self.transform = transform
        self.to_tensor = ToTensor()
    
    def __len__(self):
        """
        Len override
        :return: the number of images in a batch
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Returns an item associated with a given index
        :param idx: an index for which to return the item for
        :return: the item associated with the index
        """
        # opening images in a batch
        image_name = self.image_names[idx]
        image = Image.open(image_name)

        # applying transformations
        return self.to_tensor(image)




