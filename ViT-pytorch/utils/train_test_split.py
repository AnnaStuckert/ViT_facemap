# This code takes care of splitting the data into test and train sets, and removing NAs if there

import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# TODO there are still replicates in the test and train set
def test_train_split(csv_path, source_folder, dest_folder, train_size=0.8):
    # Ensure the base output directory exists
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Define paths for train and test CSV files
    output_path_train_csv = os.path.join(dest_folder, "train", "train_data.csv")
    output_path_test_csv = os.path.join(dest_folder, "test", "test_data.csv")

    # Ensure the train and test directories exist
    Path(os.path.dirname(output_path_train_csv)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_path_test_csv)).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)

    # Separate header and the rows starting from the 4th row
    header = data.iloc[:3]
    data_to_check = data.iloc[3:]

    # Check for rows with NA values starting from the 4th row
    na_entries = data_to_check[data_to_check.isna().any(axis=1)]
    valid_entries = data_to_check.dropna()

    # Ensure that image filenames are unique between the train and test datasets
    valid_entries_unique = valid_entries.drop_duplicates(
        subset=valid_entries.columns[2]
    )

    # Perform the split on the unique image filenames
    image_names = valid_entries_unique.iloc[:, 2].unique()
    train_images, test_images = train_test_split(
        image_names, train_size=train_size, random_state=42
    )

    train_data = valid_entries[valid_entries.iloc[:, 2].isin(train_images)]
    test_data = valid_entries[valid_entries.iloc[:, 2].isin(test_images)]

    # Concatenate header with the train and test data
    train_data_with_header = pd.concat([header, train_data])
    test_data_with_header = pd.concat([header, test_data])

    # Save the training and testing data to separate CSV files
    train_data_with_header.to_csv(output_path_train_csv, index=False)
    test_data_with_header.to_csv(output_path_test_csv, index=False)

    # Ensure the destination folders exist
    train_folder = os.path.join(dest_folder, "train")
    test_folder = os.path.join(dest_folder, "test")
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)

    # Function to copy images from a CSV to a specific folder
    def copy_from_csv(csv_path, target_folder):
        data = pd.read_csv(csv_path)
        image_names = data.iloc[:, 2].tolist()

        for image_name in image_names:
            if isinstance(image_name, str) and image_name.endswith(".png"):
                source_path = os.path.join(source_folder, image_name)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_folder)
                else:
                    print(f"Image {source_path} not found.")
            else:
                print(f"Invalid image name: {image_name}")

    # Copy training images
    copy_from_csv(output_path_train_csv, train_folder)

    # Copy testing images
    copy_from_csv(output_path_test_csv, test_folder)

    return na_entries, output_path_train_csv, output_path_test_csv
