from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize, Normalize
import os

class FacesDataset(Dataset):
    """Loads AgingGAN data"""
    def __init__(self, path:str, mode:str = "basic", size: int = 4992):
        """
        Initializes PyTorch Dataset
        :param path: path to files
        :param mode: the type of data to provide the network
        :param size: the size of the data to provide the network
        """
        # getting image names in directory
        self.young_img_names = [os.path.join(path, fname) for fname in os.listdir(path) if "jpg" in fname and int(fname.split("_")[0]) < 30][:size]
        self.old_img_names = [os.path.join(path, fname) for fname in os.listdir(path) if
                              "jpg" in fname and int(fname.split("_")[0]) > 45][:size]
        self.image_names = [os.path.join(path, fname) for fname in os.listdir(path) if "jpg" in fname][:size]

        # initializing transforms, resize, random crops
        self.to_tensor = ToTensor()
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = Resize(size=64)
        self.mode = mode

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
        if self.mode == "basic" or self.mode == "eval":
            image_name = self.image_names[idx]
            image = Image.open(image_name)

            # applying transformations
            return self.resize(self.normalize(self.to_tensor(image)))

        # adding a conditional mode
        elif self.mode == "conditional":
            image_name_young, image_name_old = self.young_img_names[idx], self.old_img_names[idx]
            image_young, image_old = Image.open(image_name_young), Image.open(image_name_old)

            # applying transformations
            return self.resize(self.normalize(self.to_tensor(image_young))), self.resize(self.normalize(self.to_tensor(image_old)))




