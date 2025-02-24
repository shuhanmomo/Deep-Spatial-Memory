import os
import sys
import gdown
import zipfile
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

sys.path.append("./helpers")
sys.path.append("./backbone")

from helpers.utils import neq_load_customized
from backbone.memdpc import MemDPC_BD
from backbone.mlp_classifier import MLPClassifier
from helpers.dataset import BldgDataset
from helpers.augmentation import (
    BrightnessJitter,
    RandomHorizontalFlip,
    Scale,
    RandomCropWithProb,
    RandomSpeedTuning,
)
from helpers.training_utils import get_data, set_path, train_one_epoch, validate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def extract_batch_features(memdpc, batch_data):
    """
    Extract features from a batch of data using the MemDPC model.

    Args:
    - memdpc (PyTorch Model): The MemDPC model instance.
    - batch_data (Tensor): A batch of data, shape (batch_size, C, H, W).

    Returns:
    - Tensor: Extracted features from the batch, shape (batch_size, feature_dim).
    """
    batch_features = []

    # Ensure the model is in evaluation mode and the batch data is on the same device as the model
    memdpc.eval()
    device = next(memdpc.parameters()).device
    batch_data = batch_data.to(device)

    # Loop over each item in the batch
    for i in range(batch_data.shape[0]):
        # Extract features for each item
        with torch.no_grad():
            context_feature = memdpc.extract_features(
                batch_data[i].unsqueeze(0)
            )  # Add batch dimension
            # Perform global average pooling
            pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
            pooled_context_feature = pooling_layer(context_feature)
            features = pooled_context_feature.view(-1, 256)
            batch_features.append(features)

    # Stack all features to form a batch
    features_batch = torch.cat(batch_features, dim=0)
    return features_batch


def calculate_accuracy(probs, labels):
    _, predicted = torch.max(probs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def train_and_test(args, epochs=50):
    # Load MemDPC
    memdpc = MemDPC_BD(
        sample_size=args["img_dim"],
        num_seq=args["num_seq"],
        seq_len=args["seq_len"],
        network=args["net"],
        pred_step=args["pred_step"],
        mem_size=args["mem_size"],
    )
    exp_path = f"log_{args['prefix']}"
    model_dir = os.listdir(exp_path)[0]
    model_dir = os.path.join(exp_path, model_dir, "model")
    model_file = ""
    for file in os.listdir(model_dir):
        if "best" in file:
            model_file = file

    model_path = os.path.join(model_dir, model_file)
    checkpoint = torch.load(model_path)
    memdpc = neq_load_customized(memdpc, checkpoint["state_dict"])
    memdpc.eval()  # MemDPC model is used only for feature extraction, not training
    mlp_model = MLPClassifier()
    mlp_model.train()

    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=0.0001)

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])
    device = torch.device("cuda")
    mlp_model.to(device)
    memdpc.to(device)

    ### data ###
    train_transform = transforms.Compose(
        [
            RandomSpeedTuning(min_dup_frames=1, max_dup_frames=5, p=args["p"]),
            RandomCropWithProb(size=[25, 50], p=args["p"], consistent=True),
            Scale(size=(30, 60)),
            RandomHorizontalFlip(consistent=True, p=args["p"]),
            BrightnessJitter(brightness=[0.5, 3], consistent=True, p=args["p"]),
        ]
    )

    val_transform = None

    train_loader = get_data(train_transform, args=args, mode="train")
    val_loader = get_data(val_transform, args=args, mode="val")
    label_map = {"room": 0, "passage": 1, "exterior": 2}
    inverse_map = {0: "room", 1: "passage", 2: "exterior"}

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    writer = SummaryWriter("mlpclasifier/1120")
    label_weights = torch.ones(len(label_map)).to(device)
    room_label_idx = label_map["room"]  # assuming 'room+passage' is a key in label_map
    label_weights[room_label_idx] = 2.0

    ### Training loop ###
    best_val_accuracy = 0
    for epoch in range(epochs):  # number of epochs
        total_train_loss = 0
        total_train_accuracy = 0
        mlp_model.train()
        step = 0
        correct_predictions_by_combined_label_train = defaultdict(int)
        total_predictions_by_combined_label_train = defaultdict(int)
        misclassified_predictions_train = defaultdict(lambda: defaultdict(int))
        print("start training")
        for batch in train_loader:
            # Extract features using MemDPC
            t_imgs = batch["t_imgs"]  # tensor of shape (num_seq, C, seq_len, H, W)
            # extract features
            features = extract_batch_features(memdpc, t_imgs)
            # Prepare labels
            start_labels = torch.tensor(
                [label_map[label] for label in batch["start_type"]]
            ).to(device)
            end_labels = torch.tensor(
                [label_map[label] for label in batch["end_type"]]
            ).to(device)
            # Forward pass through MLP
            outputs = mlp_model(features)
            start_probs, end_probs = outputs[:, :3], outputs[:, 3:]
            # Compute loss

            train_loss_start = F.cross_entropy(start_probs, start_labels)
            train_loss_end = F.cross_entropy(end_probs, end_labels)

            combined_labels = start_labels * 3 + end_labels
            combined_probs = torch.bmm(start_probs.unsqueeze(2), end_probs.unsqueeze(1))
            combined_probs = combined_probs.view(-1, 9)

            # Compute combined loss
            train_loss_combined = F.cross_entropy(combined_probs, combined_labels)
            train_loss = (
                train_loss_start + train_loss_end
            ) * 0.8 + train_loss_combined * 0.2
            total_train_loss += train_loss.item()

            # Calculate accuracy
            train_accuracy_start = calculate_accuracy(start_probs, start_labels)
            train_accuracy_end = calculate_accuracy(end_probs, end_labels)
            train_accuracy = (train_accuracy_start + train_accuracy_end) / 2
            total_train_accuracy += train_accuracy

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # logging
            writer.add_scalar(
                "Loss/train", train_loss.item(), epoch * len(train_loader) + step
            )
            writer.add_scalar(
                "Accuracy/train", train_accuracy, epoch * len(train_loader) + step
            )

            # separate label tracking
            # Predictions
            _, start_predictions = torch.max(start_probs, 1)
            _, end_predictions = torch.max(end_probs, 1)

            # Update correct and total predictions for each combined label
            for i in range(
                start_labels.size(0)
            ):  # Iterate over each instance in the batch
                start_label = batch["start_type"][i]
                end_label = batch["end_type"][i]
                actual_combined_label = start_label + "+" + end_label
                predicted_combined_label = (
                    inverse_map[int(start_predictions[i])]
                    + "+"
                    + inverse_map[int(end_predictions[i])]
                )
                correct = actual_combined_label == predicted_combined_label
                correct_predictions_by_combined_label_train[
                    actual_combined_label
                ] += int(correct)
                total_predictions_by_combined_label_train[actual_combined_label] += 1
                if actual_combined_label != predicted_combined_label:
                    misclassified_predictions_train[actual_combined_label][
                        predicted_combined_label
                    ] += 1

            step += 1
            if step % 20 == 0:
                print(
                    f"Epoch {epoch+1}, Step {step}, train_loss: {train_loss.item()}, train_acc: {train_accuracy}"
                )

        scheduler.step()
        # Validation phase
        total_val_loss = 0
        total_val_accuracy = 0
        step = 0
        mlp_model.eval()  # Set the model to evaluation mode
        correct_predictions_by_combined_label_val = defaultdict(int)
        total_predictions_by_combined_label_val = defaultdict(int)
        misclassified_predictions_val = defaultdict(lambda: defaultdict(int))
        print("start validation")
        with torch.no_grad():
            for batch in val_loader:
                # Extract features using MemDPC
                t_imgs = batch["t_imgs"]  # tensor of shape (num_seq, C, seq_len, H, W)
                # extract features
                features = extract_batch_features(memdpc, t_imgs)

                # Prepare labels

                start_labels = torch.tensor(
                    [label_map[label] for label in batch["start_type"]]
                ).to(device)
                end_labels = torch.tensor(
                    [label_map[label] for label in batch["end_type"]]
                ).to(device)

                # Forward pass through MLP
                outputs = mlp_model(features)
                start_probs, end_probs = outputs[:, :3], outputs[:, 3:]

                # Compute validation loss
                val_loss_start = F.cross_entropy(start_probs, start_labels)
                val_loss_end = F.cross_entropy(end_probs, end_labels)
                combined_labels = (
                    start_labels * 3 + end_labels
                )  # Assuming label_map has unique and sequential numeric values starting from 0
                combined_probs = torch.bmm(
                    start_probs.unsqueeze(2), end_probs.unsqueeze(1)
                )
                combined_probs = combined_probs.view(
                    -1, 9
                )  # An element-wise multiplication to get combined probabilities
                # Compute combined loss
                val_loss_combined = F.cross_entropy(combined_probs, combined_labels)
                val_loss = (
                    val_loss_start + val_loss_end
                ) * 0.8 + val_loss_combined * 0.2
                total_val_loss += val_loss.item()

                # claculate accuracy
                val_accuracy_start = calculate_accuracy(start_probs, start_labels)
                val_accuracy_end = calculate_accuracy(end_probs, end_labels)
                val_accuracy = (val_accuracy_start + val_accuracy_end) / 2
                total_val_accuracy += val_accuracy

                # logging
                writer.add_scalar(
                    "Loss/val", val_loss.item(), epoch * len(val_loader) + step
                )
                writer.add_scalar(
                    "Accuracy/val", val_accuracy, epoch * len(val_loader) + step
                )

                # separate label tracking
                # Predictions
                _, start_predictions = torch.max(start_probs, 1)
                _, end_predictions = torch.max(end_probs, 1)

                # Update correct and total predictions for each combined label
                for i in range(
                    start_labels.size(0)
                ):  # Iterate over each instance in the batch
                    start_label = batch["start_type"][i]
                    end_label = batch["end_type"][i]
                    actual_combined_label = start_label + "+" + end_label
                    predicted_combined_label = (
                        inverse_map[int(start_predictions[i])]
                        + "+"
                        + inverse_map[int(end_predictions[i])]
                    )
                    correct = actual_combined_label == predicted_combined_label
                    correct_predictions_by_combined_label_val[
                        actual_combined_label
                    ] += int(correct)
                    total_predictions_by_combined_label_val[actual_combined_label] += 1
                    if actual_combined_label != predicted_combined_label:
                        misclassified_predictions_val[actual_combined_label][
                            predicted_combined_label
                        ] += 1

                step += 1
                if step % 20 == 0:
                    print(
                        f"Epoch {epoch+1}, Step {step}, val_loss: {val_loss.item()}, val_acc: {val_accuracy}"
                    )

        # Logging epoch-level metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_model_path = "./mlpclasifier/best_model.pth"
            torch.save(mlp_model.state_dict(), best_model_path)

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", avg_train_accuracy, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val_epoch", avg_val_accuracy, epoch)
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Training Accuracy: {avg_train_accuracy}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}"
        )
        # Calculate and print accuracy for each combined label
        print("Label training accuracy breakdown:")
        for combined_label in set(total_predictions_by_combined_label_train.keys()):
            if total_predictions_by_combined_label_train[combined_label] > 0:
                accuracy = (
                    correct_predictions_by_combined_label_train[combined_label]
                    / total_predictions_by_combined_label_train[combined_label]
                )
                print(
                    f"Accuracy for combined label '{combined_label}': {accuracy * 100:.2f}%"
                )
            else:
                print(f"No predictions for combined label '{combined_label}'")
        print("Label validation accuracy breakdown:")
        for combined_label in set(total_predictions_by_combined_label_val.keys()):
            if total_predictions_by_combined_label_val[combined_label] > 0:
                accuracy = (
                    correct_predictions_by_combined_label_val[combined_label]
                    / total_predictions_by_combined_label_val[combined_label]
                )
                print(
                    f"Accuracy for combined label '{combined_label}': {accuracy * 100:.2f}%"
                )
            else:
                print(f"No predictions for combined label '{combined_label}'")
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------"
        )

    ### TEST###
    test_loader = DataLoader(BldgDataset(data_path=args["data_path"], mode="test"))
    memdpc.eval()
    mlp_model.eval()
    memdpc.to(device)
    mlp_model.to(device)

    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0

        for batch in test_loader:
            # Extract features from test data
            t_imgs = batch["t_imgs"].to(device)
            features = extract_batch_features(memdpc, t_imgs)

            # Forward pass through MLP classifier
            outputs = mlp_model(features)
            start_probs, end_probs = outputs[:, :3], outputs[:, 3:]

            # Prepare labels
            start_labels = torch.tensor(
                [label_map[label] for label in batch["start_type"]]
            ).to(device)
            end_labels = torch.tensor(
                [label_map[label] for label in batch["end_type"]]
            ).to(device)

            # Predictions
            _, start_predictions = torch.max(start_probs, 1)
            _, end_predictions = torch.max(end_probs, 1)

            # Increment correct and total predictions
            correct_predictions += (
                ((start_predictions == start_labels) & (end_predictions == end_labels))
                .sum()
                .item()
            )
            total_predictions += start_labels.size(0)

    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions
    print(
        f"Accuracy on test set (both start and end types correct): {accuracy * 100:.2f}%"
    )

    # Initialize dictionaries to track correct and total predictions for each combined label
    correct_predictions_by_combined_label = defaultdict(int)
    total_predictions_by_combined_label = defaultdict(int)
    misclassified_predictions = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in test_loader:
            # Extract features from test data
            t_imgs = batch["t_imgs"].to(device)
            features = extract_batch_features(memdpc, t_imgs)

            # Forward pass through MLP classifier
            outputs = mlp_model(features)
            start_probs, end_probs = outputs[:, :3], outputs[:, 3:]

            # Prepare labels
            start_labels = torch.tensor(
                [label_map[label] for label in batch["start_type"]]
            ).to(device)
            end_labels = torch.tensor(
                [label_map[label] for label in batch["end_type"]]
            ).to(device)

            # Predictions
            _, start_predictions = torch.max(start_probs, 1)
            _, end_predictions = torch.max(end_probs, 1)

            # Update correct and total predictions for each combined label
            for i in range(
                start_labels.size(0)
            ):  # Iterate over each instance in the batch
                start_label = batch["start_type"][i]
                end_label = batch["end_type"][i]
                actual_combined_label = start_label + "+" + end_label
                predicted_combined_label = (
                    inverse_map[int(start_predictions[i])]
                    + "+"
                    + inverse_map[int(end_predictions[i])]
                )
                correct = actual_combined_label == predicted_combined_label
                correct_predictions_by_combined_label[actual_combined_label] += int(
                    correct
                )
                total_predictions_by_combined_label[actual_combined_label] += 1
                if actual_combined_label != predicted_combined_label:
                    misclassified_predictions[actual_combined_label][
                        predicted_combined_label
                    ] += 1

    # Calculate and print accuracy for each combined label
    for combined_label in set(total_predictions_by_combined_label.keys()):
        if total_predictions_by_combined_label[combined_label] > 0:
            accuracy = (
                correct_predictions_by_combined_label[combined_label]
                / total_predictions_by_combined_label[combined_label]
            )
            print(
                f"Accuracy for combined label '{combined_label}': {accuracy * 100:.2f}%"
            )
        else:
            print(f"No predictions for combined label '{combined_label}'")

    for actual_label, predicted_labels in misclassified_predictions.items():
        print(f"Actual Label: {actual_label}")
        for predicted_label, count in predicted_labels.items():
            print(f"  Misclassified as {predicted_label}: {count} times")


def change_args(
    args,
    mem_size=128,
    epochs=75,
    batch_size=8,
    p=0.65,
    lr=5e-4,
    wd=1e-4,
    drop_out=0.3,
    data_path="./demo.zip",
    net="resnet18",
):  # try batch size
    args["mem_size"] = mem_size
    args["epochs"] = epochs
    # args['workers'] = 12
    args["batch_size"] = batch_size
    args["p"] = p
    args["lr"] = lr
    args["wd"] = wd
    args["data_path"] = data_path
    args["workers"] = 2
    args["drop_out"] = drop_out
    args["net"] = net
    return args


if __name__ == "__main__":
    DATA_PATH = "data/experiments/casestudy_1113.zip"
    args = json.load(open("./config/demo_config.json"))
    args = change_args(args, data_path=DATA_PATH)
    train_and_test()
