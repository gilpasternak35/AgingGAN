from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, RandomCrop, Resize
import os

class FacesDataset(Dataset):
    """Loads AgingGAN data"""
    def __init__(self, path, transform=None):
        # getting names in directory
        self.image_names = [os.path.join(path, fname) for fname in os.listdir(path)]
        self.transform = transform
        self.resizer = Resize(224)
        self.cropper = RandomCrop(224)
        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name)
        return self.cropper(self.resizer(self.to_tensor(image)))




