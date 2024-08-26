import os

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


# Custom transformation classes
class ZeroPadHeight(object):
    """Pad the height of the image with zeros to a given height."""

    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)  # Convert PIL Image to NumPy array

        h, w = image.shape[:2]
        new_h = self.output_height

        pad_height = max(0, new_h - h)
        top_padding = pad_height // 2
        bottom_padding = pad_height - top_padding

        image = np.pad(
            image, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode="constant"
        )

        return Image.fromarray(image)  # Convert NumPy array back to PIL Image


# Define transformation functions
def rotate_rescale(rotation, img_height, img_size):
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=rotation),
            ZeroPadHeight(img_height),
            transforms.Resize(img_size),
        ]
    )


def flip_rescale(img_height, img_size):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            ZeroPadHeight(img_height),
            transforms.Resize(img_size),
        ]
    )


def pad_rescale(img_height, img_size):
    return transforms.Compose([ZeroPadHeight(img_height), transforms.Resize(img_size)])


def rotate_flip_rescale(rotation, img_height, img_size):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=rotation),
            ZeroPadHeight(img_height),
            transforms.Resize(img_size),
        ]
    )


def blur(img_height, img_size):
    return transforms.Compose(
        [
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            ZeroPadHeight(img_height),
            transforms.Resize(img_size),
        ]
    )


# Mapping transformation names to functions
transform_options = {
    "rotate_rescale": rotate_rescale,
    "flip_rescale": flip_rescale,
    "pad_rescale": pad_rescale,
    "rotate_flip_rescale": rotate_flip_rescale,
    "blur": blur,
}


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
        # Skip the first two rows of the CSV
        self.face_frame = pd.read_csv(csv_file, skiprows=2)
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
            sample = self.__getitem__(idx, apply_transform=False)  # Get original sample
            image = sample["image"]

            for transform_name in self.transform_list:
                transform_func = transform_options[transform_name](
                    **{
                        k: self.transform_params[k]
                        for k in transform_options[transform_name].__code__.co_varnames
                        if k in self.transform_params
                    }
                )
                transformed_image = transform_func(
                    image
                )  # Apply transform directly to the image
                sample["image"] = (
                    transformed_image  # Update sample with the transformed image
                )

                d_augmented_labels = self._save_augmented_data(
                    idx, sample, transform_name
                )
                d_augmented_labels_all = pd.concat(
                    [d_augmented_labels_all, d_augmented_labels]
                )

        d_augmented_labels_all.set_index("image_name").to_csv(
            os.path.join(self.output_dir, "augmented_labels.csv")
        )

    def __getitem__(self, idx, apply_transform=True):
        # Use the third column to get image filenames
        img_filename = self.face_frame.iloc[
            idx, 2
        ]  # Adjusted to use the third column for filenames
        img_name = os.path.join(
            self.root_dir, img_filename.strip()
        )  # Strip to remove any leading/trailing whitespace

        if not os.path.exists(img_name):
            raise FileNotFoundError(f"No such file: '{img_name}'")

        try:
            image = Image.open(img_name).convert(
                "RGB"
            )  # Use PIL to open the image and convert to RGB
        except (IOError, UnidentifiedImageError) as e:
            print(f"Error opening image {img_name}: {e}")
            raise

        landmarks = self.face_frame.iloc[idx, 3:].values.reshape(
            -1, 2
        )  # Adjusted to start from the fourth column for landmarks
        sample = {"image": image, "landmarks": landmarks}

        if apply_transform and self.transform_list:
            for transform_name in self.transform_list:
                transform_func = transform_options[transform_name](
                    **{
                        k: self.transform_params[k]
                        for k in transform_options[transform_name].__code__.co_varnames
                        if k in self.transform_params
                    }
                )
                sample["image"] = transform_func(
                    sample["image"]
                )  # Apply transform to the image directly

        return sample

    def _prepare_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_augmented_data(self, idx, sample, transform_name):
        original_file_name = self.face_frame.iloc[
            idx, 2
        ]  # Adjusted to use the third column for filenames
        file_name_without_extension, file_extension = os.path.splitext(
            original_file_name
        )
        augmented_image_filename = (
            f"{file_name_without_extension}_{transform_name}_augmented.jpg"
        )
        augmented_image_path = os.path.join(self.output_dir, augmented_image_filename)

        sample["image"].save(augmented_image_path)  # Save PIL image directly

        d_augmented_labels = pd.DataFrame(sample["landmarks"].flatten()).transpose()
        d_augmented_labels.columns = self.face_frame.columns[
            3:
        ].to_list()  # Adjusted to use the appropriate landmark columns
        d_augmented_labels["image_name"] = augmented_image_filename
        return d_augmented_labels
