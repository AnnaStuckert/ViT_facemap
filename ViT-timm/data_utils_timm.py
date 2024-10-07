import logging
import os

import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image

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

        # Get image file path and load the image
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)  # Load image using skimage

        # Extract landmarks and convert them into the right format
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype(float)
        landmarks = landmarks.reshape(-1, 2)  # Ensure landmarks are reshaped

        # Create a sample dictionary
        sample = {"image": image, "landmarks": landmarks}

        # Apply any transforms (e.g., ToTensor, Normalize)
        if self.transform:
            sample = self.transform(sample)

        # Return image and landmarks separately
        return sample["image"].float(), sample["landmarks"].float()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        return {"image": image, "landmarks": torch.from_numpy(landmarks)}


class Normalize(object):
    """Normalize the image in a sample."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image = transforms.functional.normalize(image, mean=self.mean, std=self.std)
        return {"image": image, "landmarks": landmarks}


def get_loader(
    train_csv_file,
    train_data_dir,
    test_csv_file,
    test_data_dir,
    train_batch_size,
    eval_batch_size,
):
    # Dynamically load paths for training and testing data
    trainset = FaceLandmarksDataset(
        csv_file=train_csv_file,
        root_dir=train_data_dir,
        transform=transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                ToTensor(),
                # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # imagenet values
            ]
        ),
    )

    testset = FaceLandmarksDataset(
        csv_file=test_csv_file,
        root_dir=test_data_dir,
        transform=transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                ToTensor(),
                # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    train_loader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            batch_size=eval_batch_size,
            num_workers=4,
            # shuffle=True,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
