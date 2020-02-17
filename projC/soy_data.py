import torch
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os


class SoyBeanImgDataset(Dataset):
    def __init__(self, csv_file, root_dir='/Users/liam_adams/my_repos/csc591/projC/TrainData', transform=None):
        self.transform = transform
        self.img_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.img_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_frame.iloc[idx, 0])
        y = self.img_frame.iloc[idx, 1]
        image = io.imread(img_name) # all images are [480, 640, 3]
        image = np.transpose(image, (2, 0, 1))
        #image = image.astype(np.double)
        sample = {'image': image, 'y': y} # transpose gives [3, 480, 640], first conv layer wants channels first

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, y = sample['image'], sample['y']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'y': y}


def load_data(batch_size, train_split = .8):
    dataset = SoyBeanImgDataset(csv_file='/Users/liam_adams/my_repos/csc591/projC/TrainAnnotations.csv')
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, val_loader, train_size, val_size
    

def test():
    #img_data = pd.read_csv('/Users/liam_adams/my_repos/csc591/projC/TrainAnnotations.csv')
    #img_data[0]

    crop = RandomCrop(128)
    img_dataset = SoyBeanImgDataset(csv_file='/Users/liam_adams/my_repos/csc591/projC/TrainAnnotations.csv')
    i = 5
    sample = img_dataset[i]
    transformed_sample = crop(sample)
    #ax = plt.subplot(1, 3, i + 1)
    #plt.tight_layout()
    #ax.set_title(type(crop).__name__)


    #img = io.imread('TrainData/000006.jpg')
    plt.imshow(transformed_sample['image'])
    plt.show()


#test()