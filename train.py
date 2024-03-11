from torch.utils.data import DataLoader
from load_data import FacesDataset
import matplotlib.pyplot as plt

# initializing dataloader upon pytorch dataset
path_to_data = "part1"
dset = FacesDataset(path_to_data)
loader = DataLoader(dset, batch_size=4)

# printing out by batch
for batch in loader:
    print(len(batch))

    # showing image, re-permuting so that pixel channels appear first
    plt.imshow(batch[0].permute(1, 2, 0))
    plt.show()