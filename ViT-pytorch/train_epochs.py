import math
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage import io, transform
from skimage.transform import rotate
from torchvision import transforms


# Define augmentation classes
class ZeroPadHeight(object):
    """Pad the height of the image with zeros to a given height."""

    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        new_h = self.output_height

        pad_height = max(0, new_h - h)
        top_padding = pad_height // 2
        bottom_padding = pad_height - top_padding

        image = np.pad(
            image, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode="constant"
        )

        landmarks = landmarks + [0, top_padding]  # Adjust landmarks for the top padding

        return {"image": image, "landmarks": landmarks}


# Define transform functions and options
def rotate_rescale(rotation, img_height, img_size):
    class RotateRescaleTransform:
        def __init__(self):
            self.rotation = transforms.RandomRotation(degrees=rotation)
            self.zero_pad = ZeroPadHeight(img_height)
            self.resize = transforms.Resize(img_size)

        def __call__(self, sample):
            image = sample["image"]
            image = self.rotation(image)
            sample["image"] = self.resize(
                self.zero_pad({"image": image, "landmarks": sample["landmarks"]})[
                    "image"
                ]
            )
            return sample

    return RotateRescaleTransform()


def flip_rescale(img_height, img_size):
    class FlipRescaleTransform:
        def __init__(self):
            self.flip = transforms.RandomHorizontalFlip()
            self.zero_pad = ZeroPadHeight(img_height)
            self.resize = transforms.Resize(img_size)

        def __call__(self, sample):
            image = sample["image"]
            image = self.flip(image)
            sample["image"] = self.resize(
                self.zero_pad({"image": image, "landmarks": sample["landmarks"]})[
                    "image"
                ]
            )
            return sample

    return FlipRescaleTransform()


def pad_rescale(img_height, img_size):
    class PadRescaleTransform:
        def __init__(self):
            self.zero_pad = ZeroPadHeight(img_height)
            self.resize = transforms.Resize(img_size)

        def __call__(self, sample):
            sample["image"] = self.resize(self.zero_pad(sample)["image"])
            return sample

    return PadRescaleTransform()


def rotate_flip_rescale(rotation, img_height, img_size):
    class RotateFlipRescaleTransform:
        def __init__(self):
            self.flip = transforms.RandomHorizontalFlip()
            self.rotation = transforms.RandomRotation(degrees=rotation)
            self.zero_pad = ZeroPadHeight(img_height)
            self.resize = transforms.Resize(img_size)

        def __call__(self, sample):
            image = sample["image"]
            image = self.flip(image)
            image = self.rotation(image)
            sample["image"] = self.resize(
                self.zero_pad({"image": image, "landmarks": sample["landmarks"]})[
                    "image"
                ]
            )
            return sample

    return RotateFlipRescaleTransform()


def blur(img_height, img_size):
    class BlurTransform:
        def __init__(self):
            self.blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            self.zero_pad = ZeroPadHeight(img_height)
            self.resize = transforms.Resize(img_size)

        def __call__(self, sample):
            image = sample["image"]
            image = self.blur(image)
            sample["image"] = self.resize(
                self.zero_pad({"image": image, "landmarks": sample["landmarks"]})[
                    "image"
                ]
            )
            return sample

    return BlurTransform()


transform_options = {
    "rotate_rescale": rotate_rescale,
    "flip_rescale": flip_rescale,
    "pad_rescale": pad_rescale,
    "rotate_flip_rescale": rotate_flip_rescale,
    "blur": blur,
}


# Define the AugmentedFaceDataset class
class AugmentedFaceDataset:
    """Dataset class for loading, augmenting, and saving face images and their labels."""

    def __init__(
        self,
        csv_file,
        root_dir,
        output_dir,
        output_size=(846, 646),
        transform_list=None,
        transform_params=None,
    ):
        # Load the CSV file, skipping the first two columns and first two rows
        raw_data = pd.read_csv(csv_file, skiprows=2)
        self.keypoint_names = raw_data.columns[
            2:
        ].tolist()  # Use keypoint names starting from the third column
        self.face_frame = raw_data.iloc[:, 2:]  # Skip the first two columns

        self.root_dir = root_dir
        self.output_dir = output_dir
        self.output_size = output_size
        self.transform_list = transform_list  # List of transform names
        self.transform_params = (
            transform_params or {}
        )  # Dictionary of transform parameters
        self._prepare_output_directory()

    def __len__(self):
        return len(self.face_frame)

    def apply_transforms_and_save(self):
        """Apply the selected transforms and save the augmented data."""
        d_augmented_labels_all = pd.DataFrame()
        for idx in range(len(self)):
            sample = self.__getitem__(
                idx, apply_transform=True
            )  # Get original sample with transformation

            # Save augmented data
            d_augmented_labels = self._save_augmented_data(
                idx, sample, "_".join(self.transform_list)
            )
            d_augmented_labels_all = pd.concat(
                [d_augmented_labels_all, d_augmented_labels]
            )

        d_augmented_labels_all.set_index("image_name").to_csv(
            os.path.join(self.output_dir, "augmented_labels.csv")
        )

    def __getitem__(self, idx, apply_transform=True):
        # Get the filename from the third column (index 0 after skipping the first two columns)
        img_filename = self.face_frame.iloc[idx, 0]
        img_name = os.path.join(
            self.root_dir, img_filename
        )  # Construct full image path
        print(f"Loading image from: {img_name}")  # Debugging line
        image = io.imread(img_name)

        # Extract landmark coordinates, assuming they start from column 3 onwards
        landmarks = (
            self.face_frame.iloc[idx, 1:].values.astype(np.float32).reshape(-1, 2)
        )
        sample = {"image": image, "landmarks": landmarks}

        if apply_transform and self.transform_list:
            for transform_name in self.transform_list:
                transform_func = transform_options[transform_name](
                    **self.transform_params.get(transform_name, {})
                )
                sample = transform_func(sample)

        return sample

    def _prepare_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_augmented_data(self, idx, sample, transform_name):
        # Extract the original name directly from the first column of the CSV.
        original_file_name = self.face_frame.iloc[idx, 0]

        # Generate the new filenames for the augmented image and CSV by appending the transform name
        file_name_without_extension, file_extension = os.path.splitext(
            original_file_name
        )

        augmented_image_filename = (
            f"{file_name_without_extension}_{transform_name}_augmented.jpg"
        )

        # Full paths for the augmented files
        augmented_image_path = os.path.join(self.output_dir, augmented_image_filename)

        # Save the augmented image. Ensure the image data is in the correct format (e.g., scale to 255 if necessary).
        io.imsave(augmented_image_path, (sample["image"] * 255).astype(np.uint8))

        # Save the landmarks (augmented labels) in a CSV file.
        d_augmented_labels = pd.DataFrame(sample["landmarks"].flatten()).transpose()
        d_augmented_labels.columns = self.keypoint_names
        d_augmented_labels["image_name"] = augmented_image_filename
        return d_augmented_labels
