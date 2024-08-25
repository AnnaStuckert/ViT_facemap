import csv

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms  # Import torchvision transforms

from models.modeling import CONFIGS, VisionTransformer


def load_model(checkpoint_path, config_name, device):
    """
    Load the Vision Transformer model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file).
        config_name (str): Configuration name for the model.
        device (torch.device): Device to load the model on.

    Returns:
        model (torch.nn.Module): The loaded Vision Transformer model.
    """
    config = CONFIGS[config_name]
    model = VisionTransformer(
        config, num_classes=24, zero_head=False, img_size=224, vis=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame, img_size=224):
    """
    Preprocess a video frame for input to the Vision Transformer model.

    Args:
        frame (numpy.ndarray): The video frame to preprocess.
        img_size (int): Size to resize the frame to.

    Returns:
        torch.Tensor: Preprocessed frame tensor.
    """
    original_size = frame.shape[:2]  # Get original size (height, width)

    # Resize frame to match model's expected input size
    frame = cv2.resize(frame, (img_size, img_size))

    # Convert to float and normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Convert from HWC to CHW format and to tensor
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).float()

    # Define normalization transform (same as training)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Apply normalization
    frame = normalize(frame)

    # Add batch dimension
    frame = frame.unsqueeze(0)

    return frame, original_size


def run_inference_on_video(video_path, model, device):
    """
    Run inference on a video to get predicted keypoints.

    Args:
        video_path (str): Path to the input video.
        model (torch.nn.Module): The loaded Vision Transformer model.
        device (torch.device): Device to run inference on.

    Returns:
        keypoints_list (list): List of predicted keypoints for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return []

    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor, original_size = preprocess_frame(frame)  # Get original size
        frame_tensor = frame_tensor.to(device)

        with torch.no_grad():
            keypoints, _ = model(
                frame_tensor
            )  # Assuming model returns (output, attention_maps)

        keypoints = keypoints[0].cpu().numpy().flatten().tolist()

        # Scale keypoints back to original size
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
    """
    Overlay predicted keypoints on each frame of a video and save keypoints to a CSV file.

    Args:
        video_path (str): Path to the input video file.
        keypoints_list (list): List of predicted keypoints for each frame.
        output_video_path (str): Path to save the output video with keypoints overlaid.
        output_csv_path (str): Path to save the keypoints CSV file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Open CSV file to save keypoints
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

                # Save the keypoints to CSV
                writer.writerow([frame_index] + preds_keypoints)

            out.write(frame)
            frame_index += 1

    cap.release()
    out.release()
    print(f"Video with keypoints saved to {output_video_path}")
    print(f"Keypoints saved to {output_csv_path}")
