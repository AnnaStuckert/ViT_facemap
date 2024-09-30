import math
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import io, transform
from skimage.transform import rotate


def flatten(xss):
    return [x for xs in xss for x in xs]


class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.name = "rescale"

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        #if isinstance(self.output_size, int):
        #    if h > w:
        #        new_h, new_w = self.output_size * h / w, self.output_size
        #    else:
        #        new_h, new_w = self.output_size, self.output_size * w / h
        #else:
        #    new_h, new_w = self.output_size

        #new_h, new_w = int(new_h), int(new_w)

        # Ensure the final output is (224, 224)
        ##new_h = new_w = (
        #    self.output_size
        #    if isinstance(self.output_size, int)
        #    else self.output_size[0]
        #)

        #img = transform.resize(image, (new_h, new_w), anti_aliasing=True)
        img = transform.resize(image, (self.output_size,self.output_size), anti_aliasing=True)
        #landmarks = landmarks * [new_w / w, new_h / h]
        landmarks = landmarks * [self.output_size / w, self.output_size / h]
        return {"image": img, "landmarks": landmarks}


class ZeroPadHeight(object):
    """Pad the height of the image with zeros to a given height."""

    def __init__(self, output_height):
        self.output_height = output_height
        self.name = "ZeroPad"

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


class Rotate(object):
    """Rotate image and landmarks by a given angle."""

    def __init__(self, angle):
        self.angle = angle
        self.name = "rotate"

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Rotate the image
        rotated_image = rotate(
            image,
            self.angle,
            resize=False,
            center=center,
            order=1,
            mode="constant",
            cval=0,
            clip=True,
            preserve_range=False,
        )

        # Rotate the landmarks
        theta = math.radians(-self.angle)  # Convert angle from degrees to radians
        cos_angle = math.cos(theta)
        sin_angle = math.sin(theta)

        rotated_landmarks = np.empty_like(landmarks)
        for i, (x, y) in enumerate(landmarks):
            x_rotated = (
                (x - center[0]) * cos_angle - (y - center[1]) * sin_angle + center[0]
            )
            y_rotated = (
                (x - center[0]) * sin_angle + (y - center[1]) * cos_angle + center[1]
            )
            rotated_landmarks[i] = [x_rotated, y_rotated]

        return {"image": rotated_image, "landmarks": rotated_landmarks}


class GaussianBlur(object):
    """Apply Gaussian Blur to the image in the sample."""

    def __init__(self, sigma=2):
        self.sigma = sigma
        self.name = "blur"

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        blurred_image = gaussian_filter(image, sigma=self.sigma)
        return {"image": blurred_image, "landmarks": landmarks}


class HorizontalFlip(object):
    """Flip image and landmarks horizontally."""

    def __init__(self):
        self.name = "flip"

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        flipped_image = np.fliplr(image)
        image_width = image.shape[1]
        flipped_landmarks = np.copy(landmarks)
        flipped_landmarks[:, 0] = image_width - landmarks[:, 0]
        return {"image": flipped_image, "landmarks": flipped_landmarks}


class AugmentedFaceDataset:
    """Dataset class for loading, augmenting, and saving face images and their labels."""

    def __init__(
        self, csv_file, root_dir, output_dir, output_size=(846, 646), transform=None
    ):
        # Read the first two rows after the header to use as new headers
        initial_df = pd.read_csv(csv_file, nrows=2, skiprows=[0], header=None)
        new_headers = ["image_name"] + [
            f"{initial_df.iloc[0, i]}_{initial_df.iloc[1, i]}"
            for i in range(3, len(initial_df.columns))
        ]
        # Now read the CSV with the appropriate rows and columns using the new headers
        self.face_frame = pd.read_csv(
            csv_file,
            skiprows=[1, 2],  # Skip the first two rows after the header
            usecols=list(
                range(2, len(initial_df.columns))
            ),  # Use columns from the third onwards
            names=new_headers,  # Set the new header names
            header=0,  # Use the original header to ignore it in processing
        )
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.output_size = output_size
        self.transform = transform
        self._prepare_output_directory()

    def __len__(self):
        return len(self.face_frame)

    def apply_transforms_and_save(self, transforms_dict):
        """Apply a list of transforms individually and save the augmented data."""
        d_augmented_labels_all = pd.DataFrame()
        for idx in range(len(self)):
            sample = self.__getitem__(idx, apply_transform=False)  # Get original sample

            for transform_name, tsfrm in transforms_dict.items():
                # Apply transform
                print(f"Before transformation: {sample['landmarks']}")
                transformed_sample = tsfrm(sample)
                # Save augmented data using the transformation name
                d_augmented_labels = self._save_augmented_data(
                    idx, transformed_sample, transform_name
                )
                d_augmented_labels_all = pd.concat(
                    [d_augmented_labels_all, d_augmented_labels], ignore_index=True
                )
                print(f"After transformation: {sample['landmarks']}")

        # Debugging step to check if 'image_name' column exists
        print(
            "Columns in d_augmented_labels_all before setting index:",
            d_augmented_labels_all.columns,
        )

        # Ensure the output directory exists
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the consolidated CSV file
        d_augmented_labels_all.set_index("image_name").to_csv(
            os.path.join(output_dir, "augmented_labels.csv")
        )

    def __getitem__(self, idx, apply_transform=True):
        img_name = os.path.join(
            self.root_dir, self.face_frame.iloc[idx, 0]
        )  # Example image name
        image = io.imread(img_name)
        landmarks = self.face_frame.iloc[idx, 1:].values.reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform and apply_transform:
            sample = self.transform(sample)

        return sample

    def _prepare_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_augmented_data(self, idx, sample, transform_name):
        print(f"Landmarks before saving: {sample['landmarks']}")
        # Extract the original name directly from the first column of the CSV.
        original_file_name = self.face_frame.iloc[idx, 0]
        # Generate the new filenames for the augmented image and CSV by appending the transform name
        file_name_without_extension, file_extension = os.path.splitext(
            original_file_name
        )
        augmented_image_filename = (
            f"{file_name_without_extension}_{transform_name}_augmented.jpg"
        )
        # Full path for the augmented image
        augmented_image_path = os.path.join(self.output_dir, augmented_image_filename)
        # Save the augmented image
        io.imsave(augmented_image_path, (sample["image"] * 255).astype(np.uint8))
        # Prepare the DataFrame for augmented labels
        d_augmented_labels = pd.DataFrame(
            flatten(sample["landmarks"])[0:24]
        ).transpose()
        d_augmented_labels.columns = self.face_frame.columns[1:25].to_list()
        d_augmented_labels["image_name"] = augmented_image_filename
        # Debug print to check DataFrame structure
        print("DataFrame structure in _save_augmented_data:", d_augmented_labels.head())
        return d_augmented_labels
