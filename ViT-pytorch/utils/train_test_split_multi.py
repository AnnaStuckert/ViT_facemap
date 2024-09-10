import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_organize_data(base_dir, output_dir, train_size=0.8):
    # Ensure the output directory exists
    train_folder = os.path.join(output_dir, "train")
    test_folder = os.path.join(output_dir, "test")
    temp_folder = os.path.join(output_dir, "temp_renamed_images")
    output_path_train_csv = os.path.join(train_folder, "train_data.csv")
    output_path_test_csv = os.path.join(test_folder, "test_data.csv")

    # Ensure the train, test, and temp directories exist
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)
    Path(temp_folder).mkdir(parents=True, exist_ok=True)

    combined_data = pd.DataFrame()
    header = None

    # Iterate through all subfolders in the base directory
    for folder_name, subfolders, files in os.walk(base_dir):
        for file in files:
            # If a CSV file is found, process it
            if file.endswith(".csv"):
                csv_path = os.path.join(folder_name, file)
                data = pd.read_csv(csv_path)

                # Separate header and data
                if header is None:
                    header = data.iloc[:3]  # Only get the header once
                data_to_check = data.iloc[3:]

                # Check for rows with NA values and remove them
                valid_entries = data_to_check.dropna()

                # Extract the folder name (video name) to use as a prefix
                video_name = os.path.basename(folder_name)

                # Prefix image filenames with the video name (folder name) to avoid name collisions
                valid_entries.iloc[:, 2] = valid_entries.iloc[:, 2].apply(
                    lambda x: f"{video_name}_{x}" if isinstance(x, str) else x
                )

                # Append valid entries to the combined data
                combined_data = pd.concat(
                    [combined_data, valid_entries], ignore_index=True
                )

                # Copy and rename images to a temporary folder
                for image_name in valid_entries.iloc[:, 2]:
                    original_image_name = image_name.replace(f"{video_name}_", "")
                    source_image_path = os.path.join(folder_name, original_image_name)
                    if os.path.exists(source_image_path):
                        # Copy the image with the prefixed name to the temporary folder
                        dest_image_path = os.path.join(temp_folder, image_name)
                        shutil.copy(source_image_path, dest_image_path)
                    else:
                        print(f"Image {source_image_path} not found.")

    # Ensure that image filenames are unique between the train and test datasets
    valid_entries_unique = combined_data.drop_duplicates(
        subset=combined_data.columns[2]
    )

    # Perform the split on the unique image filenames
    image_names = valid_entries_unique.iloc[:, 2].unique()
    train_images, test_images = train_test_split(
        image_names, train_size=train_size, random_state=42
    )

    # Split the data into train and test sets
    train_data = combined_data[combined_data.iloc[:, 2].isin(train_images)]
    test_data = combined_data[combined_data.iloc[:, 2].isin(test_images)]

    # Concatenate header with the train and test data
    train_data_with_header = pd.concat([header, train_data])
    test_data_with_header = pd.concat([header, test_data])

    # Save the training and testing data to separate CSV files
    train_data_with_header.to_csv(output_path_train_csv, index=False)
    test_data_with_header.to_csv(output_path_test_csv, index=False)

    # Function to move renamed images to their respective folders
    def move_images(data, target_folder):
        for image_name in data.iloc[:, 2]:
            temp_image_path = os.path.join(temp_folder, image_name)
            dest_image_path = os.path.join(target_folder, image_name)
            if os.path.exists(temp_image_path):
                shutil.move(temp_image_path, dest_image_path)
            else:
                print(f"Image {temp_image_path} not found in temp folder.")

    # Move images for the training set
    move_images(train_data, train_folder)

    # Move images for the test set
    move_images(test_data, test_folder)

    # Delete the temporary folder after sorting images
    shutil.rmtree(temp_folder)

    return output_path_train_csv, output_path_test_csv
