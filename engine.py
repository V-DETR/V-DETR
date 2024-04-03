# Copyright (c) V-DETR authors. All Rights Reserved.
import torch
import datetime
import logging
import math
import time
import sys

import numpy as np
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
    batch_dict_to_cuda,
)
from utils.box_util import (flip_axis_to_camera_tensor, get_3d_box_batch_tensor)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        if args.lr_scheduler == 'cosine':
            # Cosine Learning Rate Schedule
            curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                1 + math.cos(math.pi * curr_epoch_normalized)
            )
        else:
            step_1, step_2 = args.step_epoch.split('_')
            step_1, step_2 = int(step_1), int(step_2)
            if curr_epoch_normalized < (step_1 / args.max_epoch):
                curr_lr = args.base_lr
            elif curr_epoch_normalized < (step_2 / args.max_epoch):
                curr_lr = args.base_lr / 10
            else:
                curr_lr = args.base_lr / 100
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
):
    ap_calculator = None

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device
    
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        if args.use_superpoint:
            inputs["superpoint_per_point"] = batch_data_label["superpoint_labels"]
        outputs = model(inputs)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            eta_seconds = (max_iters - curr_iter) * (time.time() - curr_time)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; ETA {eta_str}"
            )
        
        curr_iter += 1
        barrier()

    return ap_calculator, curr_iter, curr_lr, loss_avg.avg, loss_dict_reduced


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        no_nms=args.test_no_nms,
        args=args
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""


    for batch_idx, batch_data_label in enumerate(dataset_loader):   
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=net_device)
            
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        outputs = model(inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)
            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"
        else:
            loss_dict_reduced = None

        if args.cls_loss.split('_')[0] == "focalloss":
            outputs["outputs"]["sem_cls_prob"] = outputs["outputs"]["sem_cls_prob"].sigmoid()

        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        if args.axis_align_test:
            outputs["outputs"]["box_corners"] = outputs["outputs"]["box_corners_axis_align"]

        ap_calculator.step_meter(outputs, batch_data_label)
        if is_primary() and curr_iter % args.log_every == 0:
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]"
            )
        curr_iter += 1
        barrier()

    return ap_calculator, loss_avg.avg, loss_dict_reduced