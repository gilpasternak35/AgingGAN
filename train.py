from torch.utils.data import DataLoader
from load_data import FacesDataset
import matplotlib.pyplot as plt




def show_images(data_path: str) -> None:
    """
    Shows images in a given batch of data
    :param data_path: the path to the data
    :return: nothing, simply show the images
    """
    # initializing dataloader upon pytorch dataset
    dset = FacesDataset(data_path)
    loader = DataLoader(dset, batch_size=4)

    # printing out by batch
    for batch in loader:
        print(len(batch))

        # showing image, re-permuting so that pixel channels appear first
        plt.imshow(batch[0].permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    show_images("C:/Users/maxes/Downloads/part1")