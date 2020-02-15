import torch
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
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
        image = io.imread(img_name)
        sample = {'image': image, 'y': y}

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, y = sample['image'], sample['y']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'y': torch.from_numpy(y)}


def load_data():
    trainset = SoyBeanImgDataset(csv_file='/Users/liam_adams/my_repos/csc591/projC/TrainAnnotations.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader
    

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