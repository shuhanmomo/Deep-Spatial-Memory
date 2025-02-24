import os
import sys
import time
import re
import numpy as np
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import math
import zipfile
import sklearn
import shutil
from glob import glob
from PIL import Image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

sys.path.append("./helpers")
sys.path.append("./backbone")
from helpers.augmentation import (
    BrightnessJitter,
    RandomHorizontalFlip,
    Scale,
    RandomCropWithProb,
    RandomSpeedTuning,
)
from helpers.dataset import BldgDataset
from helpers.utils import (
    AverageMeter,
    save_checkpoint,
    Logger,
    neq_load_customized,
    MultiStepLR_Restart_Multiplier,
    calc_topk_accuracy,
)
from backbone.select_backbone import select_resnet
from backbone.memdpc import MemDPC_BD
from helpers.training_utils import get_data, set_path, train_one_epoch, validate

args = json.load(open("./config/demo_config.json"))


def main(args):
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])
    device = torch.device("cuda")
    num_gpu = len(str(args["gpu"]).split(","))
    print(f"num_gpu:{num_gpu}")
    # args["batch_size"] = num_gpu * args["batch_size"]

    ### model ###
    # need to test:
    # num_seq to 10?  seq_len to 3
    if args["model"] == "memdpc":
        model = MemDPC_BD(
            sample_size=args["img_dim"],
            num_seq=args["num_seq"],
            seq_len=args["seq_len"],
            network=args["net"],
            pred_step=args["pred_step"],
            mem_size=args["mem_size"],
            drop_out=args["drop_out"],
        )
    else:
        raise NotImplementedError("wrong model!")

    model.to(device)
    model = nn.DataParallel(model)
    model_without_dp = model.module

    ### optimizer ###
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args["lr"], weight_decay=args["wd"])
    criterion = nn.CrossEntropyLoss()

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

    lr_milestones_eps = [50, 100]  # can be smaller

    lr_milestones = [len(train_loader) * m for m in lr_milestones_eps]
    print(
        "=> Use lr_scheduler: %s eps == %s iters"
        % (str(lr_milestones_eps), str(lr_milestones))
    )
    lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(
        ep, gamma=0.1, step=lr_milestones, repeat=1
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0
    args["iteration"] = 1

    ### restart training ###
    if args["resume"]:
        if os.path.isfile(args["resume"]):
            print("=> loading resumed checkpoint '{}'".format(args["resume"]))
            checkpoint = torch.load(args["resume"], map_location=torch.device("cpu"))
            args["start_epoch"] = checkpoint["epoch"]
            args["iteration"] = checkpoint["iteration"]
            best_acc = checkpoint["best_acc"]
            model_without_dp.load_state_dict(checkpoint["state_dict"])
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print("[WARNING] Not loading optimizer states")
            print(
                "=> loaded resumed checkpoint '{}' (epoch {})".format(
                    args["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("[Warning] no checkpoint found at '{}'".format(args["resume"]))
            sys.exit(0)

    # logging tools
    args["img_path"], args["model_path"] = set_path(args)
    args["logger"] = Logger(path=args["img_path"])
    args["logger"].log(
        "args=\n\t\t"
        + "\n\t\t".join(["%s:%s" % (str(k), str(v)) for k, v in args.items()])
    )

    args["writer_val"] = SummaryWriter(logdir=os.path.join(args["img_path"], "val"))
    args["writer_train"] = SummaryWriter(logdir=os.path.join(args["img_path"], "train"))

    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args["start_epoch"], args["epochs"]):
        np.random.seed(epoch)
        random.seed(epoch)

        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, lr_scheduler, device, epoch, args
        )
        val_loss, val_acc = validate(val_loader, model, criterion, device, epoch, args)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_dict = {
            "epoch": epoch,
            "state_dict": model_without_dp.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "iteration": args["iteration"],
        }
        save_checkpoint(
            save_dict,
            is_best,
            filename=os.path.join(args["model_path"], "epoch%s.pth.tar" % str(epoch)),
            keep_all=False,
        )
        args["best_val_acc"] = best_acc

    print(
        "Training from ep %d to ep %d finished" % (args["start_epoch"], args["epochs"])
    )
    # sys.exit(0)

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
        args = change_args(args)
        main(args)
