import os

import pandas as pd
from PIL import Image

# Define the folder path
folder_path = "/Users/annastuckert/Documents/GitHub/ViT_facemap/ViT-pytorch/data/facemap/cam1_G7c1_1_labelled"
# Define the CSV file name
csv_filename = "cam1_G7c1_1_labels.csv"


# Function to process the content of the folder
def process_folder(folder_path, csv_filename):
    # Construct the full path to the CSV file
    csv_path = os.path.join(folder_path, csv_filename)

    # Load the CSV file
    keypoint_data = pd.read_csv(csv_path)

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter .png images
    png_files = [f for f in files if f.endswith(".png")]

    # Process each image
    for png_file in png_files:
        # Extract the base filename (without extension)
        base_filename = os.path.splitext(png_file)[0]

        # Get the keypoints for this image
        keypoints = keypoint_data[
            keypoint_data[3] == png_file
        ]  # Index 3 may have to be changed if image name is not in column 3

        if not keypoints.empty:
            # Load the image file
            png_path = os.path.join(folder_path, png_file)
            image = Image.open(png_path)

            # Process the data (here we just print the filenames and keypoint labels)
            print(f"Processing {png_file}")
            print("Keypoint Labels:")
            print(keypoints)
            print("Image Size:", image.size)
        else:
            print(f"Warning: No keypoints found for {png_file}")


# Call the function to process the folder
process_folder(folder_path, csv_filename)
