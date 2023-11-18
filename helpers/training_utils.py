from helpers.dataset import BldgDataset
import torch
import os
import time
from helpers.utils import (
    AverageMeter,
)
import numpy as np


def get_data(transform, args=None, mode="train"):
    print("Loading {} dataset for {}".format(args["dataset"], mode))

    dataset = BldgDataset(
        mode=mode,
        transform=transform,
        seq_len=args["seq_len"],
        num_seq=args["num_seq"],
        data_path=args["data_path"],
    )

    shuffle = mode == "train"
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        sampler=sampler,
        shuffle=shuffle
        and sampler
        is None,  # Only shuffle if training and no other sampler is specified
        num_workers=args["workers"],
        pin_memory=True,
        drop_last=True,  # Typically for ensuring consistent batch sizes, especially during training
    )

    print('"{}" dataset size: {}'.format(mode, len(dataset)))
    return data_loader


def set_path(args):
    if args["resume"]:
        exp_path = os.path.dirname(os.path.dirname(args["resume"]))
    else:
        exp_path = (
            f"log_{args['prefix']}/{args['model']}_{args['dataset']}-{args['img_dim'][0]}_{args['img_dim'][1]}_{args['net']}_"
            f"mem{args['mem_size']}_bs{args['batch_size']}_lr{args['lr']}_seq{args['num_seq']}_pred{args['pred_step']}_"
            f"len{args['seq_len']}_ds{args['ds']}"
        )

    img_path = os.path.join(exp_path, "img")
    model_path = os.path.join(exp_path, "model")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return img_path, model_path


def train_one_epoch(
    data_loader, model, criterion, optimizer, lr_scheduler, device, epoch, args
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = [
        [AverageMeter(), AverageMeter()],  # forward top1, top5
        [AverageMeter(), AverageMeter()],
    ]  # backward top1, top5

    model.train()
    end = time.time()
    tic = time.time()

    for idx, input_seq in enumerate(data_loader):
        # print('inside the data_loader loop now')
        data_time.update(time.time() - end)

        input_seq = input_seq["t_imgs"].to(device)

        B = input_seq.size(0)
        loss, loss_step, acc, extra = model(input_seq)

        for i in range(2):
            top1, top5 = acc[i].mean(0)  # average acc across multi-gpus
            accuracy[i][0].update(top1.item(), B)
            accuracy[i][1].update(top5.item(), B)

        loss = loss.mean()  # average loss across multi-gpus
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args["print_freq"] == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.6f}\t"
                "Acc: {acc[0][0].val:.4f}\t"
                "T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t".format(
                    epoch,
                    idx,
                    len(data_loader),
                    loss=losses,
                    acc=accuracy,
                    dt=data_time,
                    bt=batch_time,
                )
            )

            args["writer_train"].add_scalar("local/loss", losses.val, args["iteration"])
            args["writer_train"].add_scalar(
                "local/F-top1", accuracy[0][0].val, args["iteration"]
            )
            args["writer_train"].add_scalar(
                "local/F-top5", accuracy[0][1].val, args["iteration"]
            )
            args["writer_train"].add_scalar(
                "local/B-top1", accuracy[1][0].val, args["iteration"]
            )
            args["writer_train"].add_scalar(
                "local/B-top5", accuracy[1][1].val, args["iteration"]
            )

        args["iteration"] += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

    print("Epoch: [{0}]\t" "T-epoch:{t:.2f}\t".format(epoch, t=time.time() - tic))

    args["writer_train"].add_scalar("global/loss", losses.avg, epoch)
    args["writer_train"].add_scalar("global/F-top1", accuracy[0][0].avg, epoch)
    args["writer_train"].add_scalar("global/F-top5", accuracy[0][1].avg, epoch)
    args["writer_train"].add_scalar("global/B-top1", accuracy[1][0].avg, epoch)
    args["writer_train"].add_scalar("global/B-top5", accuracy[1][1].avg, epoch)

    return losses.avg, np.mean([accuracy[0][0].avg, accuracy[1][0].avg])


def validate(data_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    accuracy = [
        [AverageMeter(), AverageMeter()],  # forward top1, top5
        [AverageMeter(), AverageMeter()],
    ]  # backward top1, top5

    model.eval()

    with torch.no_grad():
        for idx, input_seq in enumerate(data_loader):
            input_seq = input_seq["t_imgs"].to(device)
            B = input_seq.size(0)
            loss, loss_step, acc, extra = model(input_seq)

            for i in range(2):
                top1, top5 = acc[i].mean(0)  # average acc across multi-gpus
                accuracy[i][0].update(top1.item(), B)
                accuracy[i][1].update(top5.item(), B)

            loss = loss.mean()  # average loss across multi-gpus
            losses.update(loss.item(), B)

    print("Validation:")

    print(
        "Epoch: [{0}/{1}]\t"
        "Loss {loss.val:.6f}\t"
        "Acc: {acc[0][0].val:.4f}\t".format(
            epoch, args["epochs"], loss=losses, acc=accuracy
        )
    )

    args["writer_val"].add_scalar("global/loss", losses.avg, epoch)
    args["writer_val"].add_scalar("global/F-top1", accuracy[0][0].avg, epoch)
    args["writer_val"].add_scalar("global/F-top5", accuracy[0][1].avg, epoch)
    args["writer_val"].add_scalar("global/B-top1", accuracy[1][0].avg, epoch)
    args["writer_val"].add_scalar("global/B-top5", accuracy[1][1].avg, epoch)

    return losses.avg, np.mean([accuracy[0][0].avg, accuracy[1][0].avg])
