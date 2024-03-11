from torch.utils.data import DataLoader
from load_data import FacesDataset
import matplotlib.pyplot as plt

path_to_data = "C:/Users/maxes/Downloads/part1"
dset = FacesDataset(path_to_data)
loader = DataLoader(dset, batch_size=4)

for batch in loader:
    print(len(batch))
    plt.imshow(batch[0].permute(1, 2, 0))
    plt.show()