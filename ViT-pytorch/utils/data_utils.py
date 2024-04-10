import logging
import pandas as pd
import torch
import os
from skimage import io, transform
import numpy as np
from torchvision.io import read_image

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler, SequentialSampler

# Specify the path to the new working directory
new_working_directory = "C:\\Users\\avs20\\Documents\\ViT-pytorch-main"

# Change the working directory
os.chdir(new_working_directory)


logger = logging.getLogger(__name__)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, csv_file, transform=None): 
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
class ZeroPadHeight(object):
    """Pad the height of the image with zeros to a given height."""

    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h = self.output_height

        pad_height = max(0, new_h - h)
        top_padding = pad_height // 2
        bottom_padding = pad_height - top_padding

        image = np.pad(image, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode='constant')

        landmarks = landmarks + [0, top_padding]  # Adjust landmarks for the top padding

        return {'image': image, 'landmarks': landmarks}
    
def get_loader(args):
    if args.dataset == "facemap":
        trainset = FaceLandmarksDataset(csv_file="./augmented_data/augmented_labels.csv",
                                           root_dir="./augmented_data/",
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
        testset = FaceLandmarksDataset(csv_file="./augmented_data_test/augmented_labels.csv",
                                           root_dir="./augmented_data_test/",
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))


    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
