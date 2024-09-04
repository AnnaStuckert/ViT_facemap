# ViT_Facemap - AVS Development Branch

![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Metrics](#metrics)
- [Acknowledgments](#acknowledgments)

## Introduction

`ViT_Facemap` is a project that utilizes Vision Transformers (ViT) for facial mapping. The project is primarily focused on the development and application of advanced machine learning models to detect and interpret facial features using ViT architectures. This repository is specifically based on the `AVS_development` branch, which includes the latest developments and experimental features.

## Features

- **Vision Transformer Integration**: Uses Vision Transformers for predicting orofacial key points.
- **DeepLabCut-based KeyPoint labelling**: The input to the ViT code is images labelled in DeepLabCut and their corresponding .csv file with keypoint coordinates. Make sure to familiarize yourself with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) and how to use their project creation and labelling sofware.
- **Customizable Pipeline**: AIM: The goal is to create code where users can easily modify and extend the pipeline to suit specific needs. TODO is to change the code to take a number of KPs (defined by the user, either manually or ideally in the config file provided by DLC in the future), and 
- **WandB**: Upon running the train_epochs script, you will be prompted with:
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
Make sure you already have a wandb account set up. press '2'.
You can find your API key in your browser here: https://wandb.ai/authorize - enter the API key in the terminal and proceed - TODO does this work the same when running in notebook?

## Usage

### Running the Model

To run the model, in the current state this entails running thebtrain_epochs.py script. This runs training and validation over epochs (adapted from steps in the original repository which this is adapted from). Currently pathways for dataloading are hardcoded into utils/data_utils.py. TODO I aim to adjust this to be adaptable.


### Arguments for `train_epochs.py`

The `train_epochs.py` script supports various arguments that control the behavior of the training process. Below is a detailed description of each argument:

#### Arguments

- `--name`: **String**  
  Default: `"test"`  
  Description: Name of this training run. Useful for monitoring and logging purposes.

- `--dataset`: **String**  
  Default: `"facemap"`  
  Description: Specifies the dataset to use for training. The default is set to `facemap`.

- `--model_type`: **String**  
  Choices: `"ViT-B_16"`, `"ViT-B_32"`, `"ViT-L_16"`, `"ViT-L_32"`, `"ViT-H_14"`, `"R50-ViT-B_16"`  
  Default: `"ViT-B_16"`  
  Description: Determines which variant of the Vision Transformer model to use.

- `--pretrained_dir`: **String**  
  Default: `"ViT-B_16.npz"`  
  Description: Path to the directory where pretrained ViT models are stored.

- `--output_dir`: **String**  
  Default: `"output"`  
  Description: The directory where model checkpoints and outputs will be saved.

- `--img_size`: **Integer**  
  Default: `224`  
  Description: Specifies the resolution size of input images.

- `--train_batch_size`: **Integer**  
  Default: `20`  
  Description: Batch size for training.

- `--eval_batch_size`: **Integer**  
  Default: `20`  
  Description: Batch size for evaluation.

- `--eval_every`: **Integer**  
  Default: `100`  
  Description: Frequency (in steps) to run evaluation on the validation set.

- `--learning_rate`: **Float**  
  Default: `2e-4`  
  Description: Initial learning rate for the optimizer.

- `--weight_decay`: **Float**  
  Default: `1e-2`  
  Description: Weight decay for regularization.

- `--num_epochs`: **Integer**  
  Default: `50`  
  Description: Total number of training epochs to run.

- `--decay_type`: **String**  
  Choices: `"cosine"`, `"linear"`  
  Default: `"linear"`  
  Description: Type of learning rate decay to use.

- `--warmup_steps`: **Integer**  
  Default: `500`  
  Description: Number of steps to perform learning rate warmup.

- `--max_grad_norm`: **Float**  
  Default: `1.0`  
  Description: Maximum gradient norm for gradient clipping.  NOTE TO SELF: not familiar with this.

- `--local_rank`: **Integer**  
  Default: `-1`  
  Description: Rank for distributed training. Set to `-1` for non-distributed training.  NOTE TO SELF: not familiar with how I should set this. I guess if I set it to 2,4,8, etc it is similar to changin batch size in DLC, but then again not sure how that would be different from using batch size here.

- `--seed`: **Integer**  
  Default: `42`  
  Description: Random seed for initialization to ensure reproducibility.

- `--gradient_accumulation_steps`: **Integer**  
  Default: `1`  
  Description: Number of steps to accumulate gradients before updating model parameters.

- `--fp16`: **Flag**  
  Default: `False`  
  Description: If set, enables 16-bit floating-point precision (mixed precision) training.  NOTE TO SELF: slightly familiar with fp precision, but could use more understanding.

- `--fp16_opt_level`: **String**  
  Default: `"O2"`  
  Description: Optimization level for mixed precision training. Options include `"O0"`, `"O1"`, `"O2"`, and `"O3"`. See [Apex AMP documentation](https://nvidia.github.io/apex/amp.html) for details.  NOTE TO SELF: not familiar with this.

- `--loss_scale`: **Float**  
  Default: `0`  
  Description: Loss scaling for improved numeric stability during mixed precision training. A value of `0` enables dynamic scaling. NOTE TO SELF: not familiar with this.

- `--root_directory`: **String**  
  Default: `"."`  
  Description: The root directory of the project, typically the folder containing `train_epochs.py`.

- `--device`: **String**  
  Choices: `"cpu"`, `"cuda"`, `"mps"`  
  Default: `"cpu"`  
  Description: Specifies the device to use for training. Can be set to `"cpu"`, `"cuda"` for Nvidia GPU acceleration, or `"mps"` for Apple Silicon devices. Beware, mps does not work well yet. When tested on M2, it was slower than using cpu.

### Example Usage

To run the `train_epochs.py` script with a specific configuration, you can use the following command:

NOT TESTED!
```bash
python train_epochs.py --name my_experiment --dataset facemap --model_type ViT-B_16 --pretrained_dir /path/to/pretrained --output_dir /path/to/output --img_size 224 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.0001 --num_epochs 30 --device cuda
```
## Metrics

This section provides an overview of the metrics used in the script to evaluate the performance of keypoint detection models. The primary metrics implemented are the Percentage of Correct Keypoints (PCK) and Root Mean Squared Error (RMSE).

### 1. Percentage of Correct Keypoints (PCK)

PCK is a metric used to measure the accuracy of keypoint predictions. It determines the percentage of keypoints that are correctly predicted within a certain threshold distance from the ground truth keypoints. The threshold is typically defined as a fraction (`alpha`) of the distance between two reference keypoints.

#### Function: `calculate_pck`

- **Purpose**: Calculates PCK for each image and identifies incorrect keypoints.
- **Arguments**:
  - `labels (pd.DataFrame)`: Ground truth keypoints with image names.
  - `preds (pd.DataFrame)`: Predicted keypoints with image names.
  - `alpha (float)`: Threshold for PCK, typically set to `0.2`.
  - `reference_points (tuple)`: Indices of keypoints to use as reference for normalization.
- **Returns**: A dictionary (`pck_results`) mapping each `image_name` to a tuple containing the PCK value and a list of incorrect keypoints.

#### How PCK is Calculated:

1. For each image, the distance between corresponding predicted and ground truth keypoints is calculated.
2. A reference distance is determined using the keypoints specified in `reference_points`.
3. Keypoints are considered correct if their prediction is within the `alpha` times the reference distance.
4. PCK is the ratio of the number of correctly predicted keypoints to the total number of keypoints.

#### Function: `average_pck`

- **Purpose**: Computes the average PCK across all images.
- **Arguments**:
  - `pck_results (dict)`: Dictionary of `image_name` to PCK values.
- **Returns**: The average PCK value as a float.

### 2. Root Mean Squared Error (RMSE)

RMSE is a common metric used to measure the average magnitude of the error between predicted and true keypoints. It gives an overall sense of how much the predictions deviate from the actual keypoints.

#### Function: `calculate_rmse`

- **Purpose**: Calculates RMSE for each image.
- **Arguments**:
  - `labels (pd.DataFrame)`: Ground truth keypoints with image names.
  - `preds (pd.DataFrame)`: Predicted keypoints with image names.
- **Returns**: A dictionary (`rmse_results`) mapping each `image_name` to its RMSE value.

#### How RMSE is Calculated:

1. For each image, the squared difference between predicted and actual keypoints is computed for all keypoints.
2. The mean of these squared differences is calculated.
3. The square root of this mean is taken to get the RMSE, representing the average error.

#### Function: `average_rmse`

- **Purpose**: Computes the average RMSE across all images.
- **Arguments**:
  - `rmse_results (dict)`: Dictionary of `image_name` to RMSE values.
- **Returns**: The average RMSE value as a float.

### Conclusion

- **PCK** provides insight into how many keypoints are correctly predicted within a specified tolerance, making it a useful metric for keypoint localization tasks. Currently it calculates how many KPs are within a distance equiivalent to 10% of the distance between KP 1 and 3 (eye top and eye bottom KPs, which I currently consider the most stable, and thus useful). This distance is scalable with mouse size, and should account for difference in subject size, thus improving from the arbitrary 4 pixel cut off used for 'accuracy' currently. However, the PCK currently does not scale with how different KPs may need a bigger or smaller margin of accaptance. e.g. if we were to label the hip, the accaptable distance to the label would be bigger than e.g. when labelling the eye (we as humans even have a harder time labelling a hip e.g.).
- **RMSE** measures the overall accuracy of keypoint predictions by calculating the average deviation, which helps in understanding the model's precision.

Ultimate average RMSE and average PCK are used as metrics.

Moreover the code currently also implementes accuracy as keypoints less then 4 pixels from the label. Currently the model is saved every time this accuracy is improved. This accuracy will be removed in favor of PCK and RMSE, and the model will be updated upon improvement in one of these metrics (to be decided)

## Acknowledgments

This repository is based on the [ViT-pytorch repository](https://github.com/jeonsworld/ViT-pytorch) by [jeonsworld](https://github.com/jeonsworld). Their work on implementing Vision Transformers in PyTorch provided a valuable foundation for this project. 

Additionally, for the implementation of attention maps based on key points (KPs), we have adapted methods from the [vit-explain repository](https://github.com/jacobgil/vit-explain) by [jacobgil](https://github.com/jacobgil). 
