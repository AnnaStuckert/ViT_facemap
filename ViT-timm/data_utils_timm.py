import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Assuming you have already defined your FaceLandmarksDataset
# from your_dataset_file import FaceLandmarksDataset


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


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)  # Use PIL to open the image
        landmarks = self.annotations.iloc[idx, 1:].values.astype(
            "float"
        )  # Assuming the landmarks are in the remaining columns

        if self.transform:
            image = self.transform(image)

        return image, landmarks


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
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet values
            ]
        ),
    )

    testset = FaceLandmarksDataset(
        csv_file=test_csv_file,
        root_dir=test_data_dir,
        transform=transforms.Compose(
            [
                ToTensor(),
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
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
