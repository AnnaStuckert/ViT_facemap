import os
import urllib.request

# Dictionary of models and their respective URLs
MODEL_URLS = {
    "R50+ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/R50%2BViT-B_16.npz",
    "ViT-B_16-224": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz",
    "ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz",
    "ViT-B_32": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz",
    "ViT-B_8": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_8.npz",
    "ViT-L_16-224": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz",
    "ViT-L_16": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz",
    "ViT-L_32": "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz",
}


def download_model(model_name, root_directory="."):
    # Check if the model name is in the available model URLs
    if model_name not in MODEL_URLS:
        raise ValueError(
            f"Model {model_name} is not available. Available models: {list(MODEL_URLS.keys())}"
        )

    # Define the folder name and the model URL
    folder_name = "model_files"
    model_url = MODEL_URLS[model_name]

    # Create the full path for the folder
    folder_path = os.path.join(root_directory, folder_name)

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the path for the model file (use the model name to generate the filename)
    model_path = os.path.join(folder_path, model_name + ".npz")

    # Download the model if it doesn't already exist
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} to {folder_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    else:
        print(f"Model {model_name} already exists in {folder_path}.")

    # Return the path to the downloaded model
    return model_path
