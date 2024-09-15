# coding=utf-8
from __future__ import absolute_import, division, print_function

# Import necessary modules
import argparse  # This line was missing
import csv
import logging
import os
import random
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from apex import amp  # For running on a GPU
from apex.parallel.distributed import DistributedDataParallel as DDP
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.performance_metrics import (
    average_pck,
    average_rmse,
    calculate_pck,
    calculate_rmse,
)
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule


def save_predictions_to_csv(predictions, filepath):
    """
    Saves the predicted keypoints to a CSV file.

    Args:
        predictions (numpy.ndarray): The predictions to save.
        filepath (str): The path to the file where predictions should be saved.
    """
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageID", "Predictions"])  # Writing header
        for i, pred in enumerate(predictions):
            writer.writerow([i, " ".join(map(str, pred))])


logger = logging.getLogger(__name__)


class LossCurve(object):
    def __init__(self):
        self.d_lossCurve = {
            "epoch": [],  # Added epoch field
            "steps": [],
            "metric": [],
            "training_loss": [],
            "validation_loss": [],
            "validation_acc": [],
        }

    def update(
        self,
        epoch,
        step,
        metric,
        training_loss,
        validation_loss=None,
        validation_acc=None,
    ):
        self.d_lossCurve["epoch"].append(epoch)  # Record the epoch
        self.d_lossCurve["steps"].append(step)
        self.d_lossCurve["metric"].append(metric)
        self.d_lossCurve["training_loss"].append(training_loss)
        if validation_loss is not None:
            self.d_lossCurve["validation_loss"].append(validation_loss)
        else:
            self.d_lossCurve["validation_loss"].append(None)
        if validation_acc is not None:
            self.d_lossCurve["validation_acc"].append(validation_acc)
        else:
            self.d_lossCurve["validation_acc"].append(None)

    def save(self, fileName, args):
        file_path = os.path.join(args.output_dir, fileName)
        df = pd.DataFrame(self.d_lossCurve)
        df.to_csv(file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, (int, float)):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    diff = preds - labels
    diff_abs = np.abs(diff)
    return (
        diff_abs < 4
    ).mean()  # Consider if accuracy can be refined so different cut offs for different points.


def save_model(args, model, epoch):
    model_to_save = model.module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(
        args.output_dir, f"{args.name}_checkpoint_epoch_{epoch}.pth"
    )
    # model_checkpoint = os.path.join(args.output_dir, f"{args.name}_testing20240907.pth") for saving just one model

    torch.save(
        {"state_dict": model_to_save.state_dict(), **vars(args)}, model_checkpoint
    )
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

    # upload model to wandb
    wandb.save(model_checkpoint)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    num_KPs = args.n_KPs * 2  # Number of keypoints to be tracked * 2 for xy coordinates

    model = VisionTransformer(config, args.img_size, zero_head=True, num_KPs=num_KPs)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    num_KPs = args.n_KPs * 2
    # Validation
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0],
    )
    loss_fct = torch.nn.MSELoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(batch[t].to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            y = y.view(y.shape[0], num_KPs)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

        if len(all_preds) == 0:
            all_preds.append(logits.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], logits.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    loss = eval_losses.avg

    d_preds = pd.DataFrame(all_preds)
    d_labels = pd.DataFrame(all_label)

    image_names = pd.read_csv(args.test_csv_file)

    d_preds["image_names"] = image_names["image_name"]
    d_labels["image_names"] = image_names["image_name"]
    print(d_preds)
    print(d_labels)

    pck_results = calculate_pck(
        labels=d_labels, preds=d_preds, alpha=0.1, reference_points=(1, 3)
    )  # Adjust alpha and reference points as needed - this should be part of the input argument
    avg_pck = average_pck(pck_results)

    rmse_results = calculate_rmse(labels=d_labels, preds=d_preds)
    avg_rmse = average_rmse(rmse_results)

    predictions_csv = "predictions.csv"
    labels_csv = "labels.csv"

    # Combine the output directory with the file names
    predictions_csv_path = os.path.join(args.output_dir, predictions_csv)
    labels_csv_path = os.path.join(args.output_dir, labels_csv)

    # Save to CSV in the specified output directory
    d_preds.to_csv(predictions_csv_path)
    d_labels.to_csv(labels_csv_path)

    # Log to W&B
    wandb.save(predictions_csv)  # Save predictions.csv to W&B
    wandb.save(labels_csv)  # Optionally save labels.csv to W&B

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    logger.info("Average PCK: %2.5f" % avg_pck)
    logger.info("Average RMSE: %2.5f" % avg_rmse)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    writer.add_scalar("test/avg_pck", scalar_value=avg_pck, global_step=global_step)
    writer.add_scalar("test/avg_rmse", scalar_value=avg_rmse, global_step=global_step)

    return accuracy, loss, avg_pck, avg_rmse


def train(args, model):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Calculate total number of training steps
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )
    else:
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )

    if args.fp16:
        model, optimizer = amp.initialize(
            models=model, optimizers=optimizer, opt_level=args.fp16_opt_level
        )
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.local_rank != -1:
        model = DDP(
            model, message_size=250000000, gradient_predivide_factor=get_world_size()
        )

    logger.info("***** Running training *****")
    logger.info("  Num epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility
    losses = AverageMeter()
    lossCurve = LossCurve()
    global_step, best_acc, best_rmse, best_loss = 0, 0, 1000000, 1000000

    for epoch in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1} of {args.num_epochs} (loss={losses.val})",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=args.local_rank not in [-1, 0],
        )

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(batch[t].to(args.device) for t in batch)
            x, y = batch
            loss = model.forward(x.float(), y.float())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    f"Training Epoch {epoch + 1} (global_step={global_step}, loss={losses.val:.5f})"
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar(
                        "train/loss", scalar_value=losses.val, global_step=global_step
                    )
                    writer.add_scalar(
                        "train/lr",
                        scalar_value=scheduler.get_lr()[0],
                        global_step=global_step,
                    )

        # Run validation and log results at the end of the epoch
        if args.local_rank in [-1, 0]:
            accuracy, loss_valid, avg_pck, avg_rmse = valid(
                args, model, writer, test_loader, global_step
            )
            lossCurve.update(epoch + 1, global_step, "training_loss", losses.avg)
            lossCurve.update(epoch + 1, global_step, "validation_loss", loss_valid)
            lossCurve.update(epoch + 1, global_step, "validation_acc", accuracy)
            lossCurve.update(epoch + 1, global_step, "avg_pck", avg_pck)
            lossCurve.update(epoch + 1, global_step, "avg_rmse", avg_rmse)

            # Log validation metrics with W&B
            wandb.log(
                {
                    "validation_loss": loss_valid,
                    "validation_accuracy": accuracy,
                    "average_pck": avg_pck,
                    "average_rmse": avg_rmse,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                }
            )

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_model(args, model, epoch)
            # TODO update model saving not using accuracy but PCK or RMSE
            if (
                best_acc < accuracy
            ):  # accuracy should be higher than the existing accuracy to save
                # save_model(args, model)
                best_acc = accuracy
            if best_loss > loss_valid:
                best_loss = loss_valid
            # this should replace best_acc when I am sure avg_rmse is calculated over entire validation, not just last instance
            if best_rmse > avg_rmse:
                # save_model(args, model, epoch)
                best_rmse = avg_rmse
            model.train()

        # Save loss curve after each epoch
        lossCurve.save("lossCurve.csv", args)
        losses.reset()

    if args.local_rank in [-1, 0]:
        writer.close()

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best Loss (MSE): \t%f" % best_loss)
    logger.info("End Training!")


def main():
    # Get the path of the current script (train file)
    current_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name",
        default="facemap_ViT_ResNet",
        help="Name of this run. Used for monitoring.",
    )
    parser.add_argument("--dataset", default="facemap", help="Which downstream task.")
    parser.add_argument(
        "--model_type",
        choices=[
            "ViT-B_16",
            "ViT-B_32",
            "ViT-L_16",
            "ViT-L_32",
            "ViT-H_14",
            "R50-ViT-B_16",
        ],
        # default="ViT-B_16",
        default="R50-ViT-B_16",
        help="Which variant to use.",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        # default=os.path.join(current_dir, "model_files", "ViT-B_16.npz"),
        default=os.path.join(current_dir, "model_files", "R50+ViT-B_16.npz"),
        help="Path to the pretrained ViT model file in the model_files directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(current_dir, "projects/Facemap/output"),
        type=str,
        help="The output directory where checkpoints will be written.",
    )
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument(
        "--train_batch_size",
        default=20,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=20, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--eval_every",  # is the relevant anymore if I evaluate after each epoch?
        default=100,
        type=int,
        help="Run prediction on validation set every so many steps."
        "Will always run one evaluation at the end of training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--num_epochs",
        default=300,  # Changed from num_steps to num_epochs
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        default="linear",
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on GPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--root_directory",
        type=str,
        default=".",
        help="Location of root directory (currently the ViT_pytorch folder)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Device to use for training: 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument(
        "--train_csv_file",
        type=str,
        # required=True,
        default=os.path.join(
            current_dir,
            "projects/Facemap/data/train/augmented_data/augmented_labels.csv",
        ),
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        # required=True,
        default=os.path.join(current_dir, "projects/Facemap/data/train/augmented_data"),
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--test_csv_file",
        type=str,
        # required=True,
        default=os.path.join(
            current_dir,
            "projects/Facemap/data/test/augmented_data/augmented_labels.csv",
        ),
        help="Path to the testing CSV file.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        # required=True,
        default=os.path.join(current_dir, "projects/Facemap/data/test/augmented_data"),
        help="Directory containing testing images.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        # required=True,
        default="facemap_project",
        help="Name for project on wandb",
    )
    parser.add_argument(
        "--n_KPs",
        type=int,
        default=12,
        help="Number of keypoints to predict. This will be multiplied with 2 in the algorithm to account for x,y coordinates of each KP. Thus enter the number of KPs where both x and y coordinate is accounted for so 12 KP = 24 coodinates, x and y for each KP, then enter 12",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save the model every X epochs. Set to 0 to disable saving.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Whether to use Weights & Biases for logging.",
    )
    args = parser.parse_args()

    # Save arguments to a config file for model specification
    config_filename = f"config_{args.name}.yaml"
    import yaml

    # Combine output directory with the config filename
    config_filepath = os.path.join(args.output_dir, config_filename)

    # Ensure the output directory exists
    if not os.path.exists(os.path.dirname(config_filepath)):
        os.makedirs(os.path.dirname(config_filepath))

    # Save the configuration file in the specified output directory
    with open(config_filepath, "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    # Setup device: CUDA, MPS (Apple Silicon), or CPU
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            args.n_gpu = torch.cuda.device_count()
        else:
            print("CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
            args.n_gpu = 0
    elif args.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            args.n_gpu = 1
        else:
            print("MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
            args.n_gpu = 0
    elif args.device == "cpu":
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        print(f"Unrecognized device type '{args.device}'. Falling back to CPU.")
        device = torch.device("cpu")
        args.n_gpu = 0

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s"
        % (
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )
    )

    # Initialize W&B if specified
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            config=vars(args),
            dir=args.output_dir,
        )

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)

    # Finish wandb logging
    wandb.finish()


if __name__ == "__main__":
    main()
