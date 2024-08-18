# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from apex import amp  # I think these are for running on a GPU
from apex.parallel.distributed import DistributedDataParallel as DDP
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

# Specify the path to the new working directory
# new_directory = "C:\\Users\\avs20\\Documents\\Github\\ViT_facemap\\ViT-pytorch"
# new_directory = "/Users/annastuckert/Documents/GitHub/ViT_facemap/ViT-pytorch"


# Change the working directory
# os.chdir(new_directory)


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
            "steps": [],
            "metric": [],
            "training_loss": [],
            "validation_loss": [],
            "validation_acc": [],
        }

    def update(
        self, step, metric, training_loss, validation_loss=None, validation_acc=None
    ):
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

    def save(self, fileName):
        df = pd.DataFrame(self.d_lossCurve)
        df.to_csv(fileName)

    def plot(self):
        print("Data:", self.d_lossCurve)  # Debug print statement
        colors = {
            "training_loss": "blue",
            "validation_loss": "orange",
            "validation_acc": "green",
        }  # Define colors for different metrics
        for metric, color in colors.items():
            indices = [
                i for i, m in enumerate(self.d_lossCurve["metric"]) if m == metric
            ]
            print(f"Metric: {metric}, Indices: {indices}")  # Debug print statement
            if indices:
                steps = [self.d_lossCurve["steps"][i] for i in indices]
                loss = [self.d_lossCurve[metric][i] for i in indices]
                print(f"Steps: {steps}, Loss: {loss}")  # Debug print statement
                plt.plot(steps, loss, label=f"{metric.capitalize()} Loss", color=color)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()


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
    ).mean()  # Currently the model will plateau at a MSE loss around 20 whe  training with the current parameters for 2000 epochs. the square root of that is ~4-5, so I'll open up some pictures and sanity check that the actual point I am looking for is I would allow 5 ish pixels from the actual label, and still consider it a hit, so the cut off is informed by both the MSE loss from the model, and visual ispection and sanity check. Consider if accuracy can be refined so different cut offs for different points, e.g. maybe more leniency for mouth point than eye corner points.


def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    # Adjust the filename format if needed
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)

    # Save the model state dictionary along with other necessary parameters using torch.save()
    torch.save(
        {"state_dict": model_to_save.state_dict(), **vars(args)}, model_checkpoint
    )

    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 24  # var 24

    model = VisionTransformer(
        config, args.img_size, zero_head=True, num_classes=num_classes
    )
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
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
    # Validation!
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
        batch_1 = tuple(batch[t].to(args.device) for t in batch)
        x, y = batch_1
        with torch.no_grad():
            logits = model(x)[0]
            y = y.view(y.shape[0], 24)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(
                logits, dim=-1
            )  # think this can be commented out, should not be needed since we are not doing classification but regression

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

    image_names = pd.read_csv("augmented_data_test/augmented_labels.csv")

    d_preds["image_names"] = image_names["image_name"]
    d_labels["image_names"] = image_names["image_name"]

    d_preds.to_csv("predictions.csv")
    d_labels.to_csv("labels.csv")

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy, loss


def train(args, model):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler - adam is more robust, less sensitive to hyperparameters, so if initial learning rate is off adam handles it better
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    t_total = args.num_steps
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

    # Distributed training
    if args.local_rank != -1:
        model = DDP(
            model, message_size=250000000, gradient_predivide_factor=get_world_size()
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    lossCurve = LossCurve()
    global_step, best_acc, best_loss = 0, 0, 1000000
    first_eval_done = False  # Track if the first evaluation has been done

    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training ({global_step} / {t_total} Steps) (loss={losses.val})",
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
                lossCurve.update(global_step, "training_loss", loss.item())
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
                    "Training (%d / %d Steps) (loss=%2.5f)"
                    % (global_step, t_total, losses.val)
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
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, loss_valid = valid(
                        args, model, writer, test_loader, global_step
                    )
                    lossCurve.update(global_step, "validation_loss", loss_valid)
                    lossCurve.update(global_step, "validation_acc", accuracy)
                    # Save the model after the first evaluation
                    if not first_eval_done:
                        save_model(args, model)
                        first_eval_done = True
                    # going forward, save the model every time accuracy improves
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    if best_loss > loss_valid:
                        best_loss = loss_valid
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()

    lossCurve.save("lossCurve.csv")
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best Loss (MSE): \t%f" % best_loss)
    logger.info("End Training!")
    lossCurve.plot()


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name", default="test", help="Name of this run. Used for monitoring."
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
        default="ViT-B_16",
        help="Which variant to use.",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="ViT-B_16.npz",
        help="Where to search for pretrained ViT models.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
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
        "--eval_every",
        default=10,
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
        "--weight_decay", default=1e-2, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--num_steps",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )  # plateaus loss at around 1500 steps at current other
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        default="linear",  # changed from cosine as I believe this is what Yichen did
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=1,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )  # was 500
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,  # tried adjusting this from 1 to 25 to match Yichen
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
    args = parser.parse_args()

    # Save argusments to a config file for model specification
    config_filename = f"config_{args.name}.yaml"

    import yaml

    with open(config_filename, "w") as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(minutes=60)
        )
        args.n_gpu = 1
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

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
