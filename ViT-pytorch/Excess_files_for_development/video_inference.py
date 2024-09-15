import csv
import glob
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.modeling import CONFIGS, VisionTransformer


def load_model(checkpoint_path, config_name, device):
    config = CONFIGS[config_name]
    model = VisionTransformer(
        config, num_KPs=24, zero_head=False, img_size=224, vis=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame, img_size=224):
    original_size = frame.shape[:2]
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32) / 255.0
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).float()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frame = normalize(frame)
    frame = frame.unsqueeze(0)
    return frame, original_size


def run_inference_on_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return []

    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor, original_size = preprocess_frame(frame)
        frame_tensor = frame_tensor.to(device)

        with torch.no_grad():
            keypoints, _ = model(frame_tensor)

        keypoints = keypoints[0].cpu().numpy().flatten().tolist()

        scaled_keypoints = []
        for i in range(0, len(keypoints), 2):
            x_scaled = keypoints[i] * (original_size[1] / 224.0)
            y_scaled = keypoints[i + 1] * (original_size[0] / 224.0)
            scaled_keypoints.extend([x_scaled, y_scaled])

        keypoints_list.append(scaled_keypoints)

    cap.release()
    return keypoints_list


def overlay_keypoints_on_video_and_save_csv(
    video_path, keypoints_list, output_video_path, output_csv_path
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Keypoints"])

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index < len(keypoints_list):
                preds_keypoints = keypoints_list[frame_index]
                for i in range(0, len(preds_keypoints), 2):
                    cv2.circle(
                        frame,
                        (int(preds_keypoints[i]), int(preds_keypoints[i + 1])),
                        4,
                        (0, 255, 255),
                        -1,
                    )

                writer.writerow([frame_index] + preds_keypoints)

            out.write(frame)
            frame_index += 1

    cap.release()
    out.release()
    print(f"Video with keypoints saved to {output_video_path}")
    print(f"Keypoints saved to {output_csv_path}")


# Example execution

from pathlib import Path

# Import necessary functions and libraries
import torch

from Excess_files_for_development.video_inference import (
    load_model,
    overlay_keypoints_on_video_and_save_csv,
    run_inference_on_video,
)

# Define paths and configuration
video_path = "/Users/annastuckert/Documents/GitHub/ViT_facemap/ViT-pytorch/Facemap_videos/cam1_G7c1_1_10seconds.avi"  # Path to your input video
# checkpoint_path = '/Users/annastuckert/Documents/GitHub/ViT_facemap/ViT-pytorch/model_checkpoints/test_checkpoint.pth'  # Path to your model checkpoint file
checkpoint_path = (
    Path("projects")
    / "Facemap"
    / "wandb_model"
    / "output"
    / "facemap_with_augmentation_300epochs_checkpoint_epoch_299.pth"
)
output_video_path = "output/keypoints.mp4"  # Path to save the output video
output_csv_path = "output/keypoints.csv"  # Path to save the keypoints CSV file
config_name = "ViT-B_16"  # Use the appropriate configuration name for your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device

# Load the model
model = load_model(checkpoint_path, config_name, device)

# Run inference on the video to get predicted keypoints
keypoints_list = run_inference_on_video(video_path, model, device)

# Overlay the predicted keypoints on the video frames and save the output
overlay_keypoints_on_video_and_save_csv(
    video_path, keypoints_list, output_video_path, output_csv_path
)

# Output paths and check files
print(f"Output video saved to: {output_video_path}")
print(f"Output CSV saved to: {output_csv_path}")
