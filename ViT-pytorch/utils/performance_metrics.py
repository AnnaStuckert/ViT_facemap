import numpy as np
import pandas as pd


def calculate_pck(labels, preds, alpha=0.2, reference_points=(4, 5)):
    """
    Calculate PCK (Percentage of Correct Keypoints) for each image and identify incorrect keypoints.

    Args:
        labels (pd.DataFrame): Ground truth keypoints with image names.
        preds (pd.DataFrame): Predicted keypoints with image names.
        alpha (float): Threshold for PCK, typically 0.2.
        reference_points (tuple): Indices of keypoints to use as reference for normalization.

    Returns:
        pck_results (dict): Dictionary of image_name to (PCK value, list of incorrect keypoints).
    """
    pck_results = {}

    # Number of keypoints
    num_keypoints = 12  # 24 / 2
    # works in the csv files
    # keypoints_columns = list(range(1, 25))  # Columns from index 1 to 24

    # maybe works in train_epochs
    keypoints_columns = list(range(0, 24))  # Columns from index 1 to 24

    # Check for correct format
    if "image_names" not in labels.columns or "image_names" not in preds.columns:
        raise ValueError(
            "Both labels and predictions must contain 'image_names' column."
        )

    if len(reference_points) != 2 or any(x >= num_keypoints for x in reference_points):
        raise ValueError("Reference points must be valid keypoint indices.")

    for _, row in labels.iterrows():
        image_name = row["image_names"]
        # print(keypoints_columns)
        keypoints_data = row[keypoints_columns].values

        try:
            label_keypoints = keypoints_data.reshape(
                num_keypoints, 2
            )  # Reshape into (12, 2)
        except ValueError as e:
            print(f"Error reshaping data for image {image_name}: {e}")
            continue

        pred_row = preds.loc[preds["image_names"] == image_name]
        if pred_row.empty:
            print(f"No predictions found for image {image_name}")
            continue
        pred_keypoints_data = pred_row.iloc[0, keypoints_columns].values

        try:
            pred_keypoints = pred_keypoints_data.reshape(
                num_keypoints, 2
            )  # Reshape into (12, 2)
        except ValueError as e:
            print(f"Error reshaping prediction data for image {image_name}: {e}")
            continue

        # Calculate reference distance
        ref_distance = np.linalg.norm(
            label_keypoints[reference_points[0]] - label_keypoints[reference_points[1]]
        )
        threshold_distance = alpha * ref_distance

        correct_keypoints = 0
        incorrect_keypoints = []

        for j in range(num_keypoints):
            distance = np.linalg.norm(pred_keypoints[j] - label_keypoints[j])
            if distance < threshold_distance:
                correct_keypoints += 1
            else:
                incorrect_keypoints.append(j)

        pck = correct_keypoints / num_keypoints
        pck_results[image_name] = (pck, incorrect_keypoints)

    return pck_results


def average_pck(pck_results):
    """
    Calculate the average PCK from the PCK results.

    Args:
        pck_results (dict): Dictionary of image_name to PCK values.

    Returns:
        avg_pck (float): Average PCK across all images.
    """
    total_pck = sum(pck[0] for pck in pck_results.values())
    num_images = len(pck_results)
    avg_pck = total_pck / num_images if num_images > 0 else 0
    return avg_pck


def calculate_rmse(labels, preds):
    """
    Calculate RMSE (Root Mean Squared Error) for each image.

    Args:
        labels (pd.DataFrame): Ground truth keypoints with image names.
        preds (pd.DataFrame): Predicted keypoints with image names.

    Returns:
        rmse_results (dict): Dictionary of image_name to RMSE value.
    """
    rmse_results = {}

    # Number of keypoints
    num_keypoints = 12  # 24 / 2
    # keypoints_columns = list(range(1, 25))  # Columns from index 1 to 24
    keypoints_columns = list(range(0, 24))  # Columns from index 1 to 24

    # Check for correct format
    if "image_names" not in labels.columns or "image_names" not in preds.columns:
        raise ValueError(
            "Both labels and predictions must contain 'image_names' column."
        )

    for _, row in labels.iterrows():
        image_name = row["image_names"]
        keypoints_data = row[keypoints_columns].values

        try:
            label_keypoints = keypoints_data.reshape(
                num_keypoints, 2
            )  # Reshape into (12, 2)
        except ValueError as e:
            print(f"Error reshaping data for image {image_name}: {e}")
            continue

        pred_row = preds.loc[preds["image_names"] == image_name]
        if pred_row.empty:
            print(f"No predictions found for image {image_name}")
            continue
        pred_keypoints_data = pred_row.iloc[0, keypoints_columns].values

        try:
            pred_keypoints = pred_keypoints_data.reshape(
                num_keypoints, 2
            )  # Reshape into (12, 2)
        except ValueError as e:
            print(f"Error reshaping prediction data for image {image_name}: {e}")
            continue

        # Calculate squared errors for RMSE
        squared_errors = [
            np.linalg.norm(pred_keypoints[j] - label_keypoints[j]) ** 2
            for j in range(num_keypoints)
        ]
        rmse = np.sqrt(np.mean(squared_errors))
        rmse_results[image_name] = rmse

    return rmse_results


def average_rmse(rmse_results):
    """
    Calculate the average RMSE from the RMSE results.

    Args:
        rmse_results (dict): Dictionary of image_name to RMSE values.

    Returns:
        avg_rmse (float): Average RMSE across all images.
    """
    total_rmse = sum(rmse_results.values())
    num_images = len(rmse_results)
    avg_rmse = total_rmse / num_images if num_images > 0 else 0
    return avg_rmse
