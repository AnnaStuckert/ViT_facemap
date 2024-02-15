import logging
import pandas as pd
import torch
import os
from skimage import io
import numpy as np
from torchvision.io import read_image

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler, SequentialSampler



logger = logging.getLogger(__name__)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,  root_dir, csv_file, transform=None, target_transform=None): 
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

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 2])
#        image = io.imread(img_name)
        image = read_image(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 3:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

#        if self.transform:
#            sample = self.transform(sample)

        return sample
#        if self.transform:
 #           image = self.transform(image)
  #      if self.target_transform:
   #         label = self.target_transform(landmarks)
    #    return image, landmarks


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "facemap":
        trainset = FaceLandmarksDataset(root_dir="./data/facemap/Train/",
                                        csv_file='./data/facemap/Train/train.csv',
                                    transform=transform_train)
        testset = FaceLandmarksDataset(root_dir="./data/facemap/Test/",
                                       csv_file='./data/facemap/Test/test.csv',
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    train_loader = DataLoader(trainset,
#                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
#                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
