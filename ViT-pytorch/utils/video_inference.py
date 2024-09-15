import csv
import glob
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.modeling import CONFIGS, VisionTransformer


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

    print(
        f"Initializing VideoWriter for {output_video_path} with resolution ({width}, {height})"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Ensure the directory for CSV output exists
    csv_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(csv_dir):
        print(f"Creating directory: {csv_dir}")
        os.makedirs(csv_dir)

    print(f"Opening CSV file: {output_csv_path}")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Keypoints"])

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or error reading frame {frame_index}")
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
