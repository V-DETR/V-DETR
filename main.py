# Copyright (c) V-DETR authors. All Rights Reserved.
import argparse
import os
import sys
import pickle

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler
import MinkowskiEngine as ME

from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible

import wandb


def wandb_log(*args, **kwargs):
    if is_primary():
        wandb.log(*args, **kwargs)
        

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=7e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="vdetr",
        type=str,
        help="Name of the model",
    )
    parser.add_argument("--num_points", default=100000, type=int)
    
    parser.add_argument("--minkowski", default=True, action="store_true")
    parser.add_argument("--mink_syncbn", default=True, action="store_true")
    parser.add_argument("--stem_bn", default=True, action="store_true")
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--depth", default=34, type=int)
    parser.add_argument("--inplanes", default=64, type=int)
    parser.add_argument("--num_stages", default=4, type=int)
    parser.add_argument("--use_fpn", default=True, action="store_true")
    parser.add_argument("--layer_idx", default=0, type=int)
    parser.add_argument("--no_mink_first_pool", default=True, action="store_true")
    parser.add_argument("--enc_dim", default=256, type=int)
    
    ### Encoder #we remove the transformer encoder of 3DETR, which is not necessary
    # parser.add_argument(
    #     "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    # ) 
 
    ### Decoder
    parser.add_argument("--dec_nlayers", default=9, type=int) #note: the light FFN which is regarded as a simple decoder layer
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    parser.add_argument("--rpe_dim", default=128, type=int)
    parser.add_argument("--rpe_quant", default="bilinear_4_10", type=str)
    parser.add_argument("--log_scale", default=512, type=float)
    
    parser.add_argument("--pos_for_key", default=False, action="store_true")
    parser.add_argument("--querypos_mlp", default=True, action="store_true")
    parser.add_argument("--q_content", default="random", type=str)
    
    parser.add_argument("--repeat_num", default=5, type=int) # if you want use the model without NMS, please set the repeat_num==0
    parser.add_argument("--proj_nohid", default=True, action="store_true")
    parser.add_argument("--woexpand_conv", default=True, action="store_true")
    parser.add_argument("--share_selfattn", default=False, action="store_true")

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument("--mlp_norm", default="bn1d", type=str)
    parser.add_argument("--mlp_act", default="relu", type=str)
    parser.add_argument("--mlp_sep", default=True, action="store_true")
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=4096, type=int)
    parser.add_argument("--nqueries", default=1024, type=int)
    parser.add_argument("--is_bilable", default=True, action="store_true")
    parser.add_argument("--no_first_repeat", default=True, action="store_true") 
    parser.add_argument("--use_superpoint", default=False, action="store_true")
    parser.add_argument("--axis_align_test", default=False, action="store_true")
    parser.add_argument("--iou_type", default="giou", choices=['giou','diou', 'iou'], type=str)
    parser.add_argument("--angle_type", default="", type=str, choices=['world_coords', 'object_coords'], help="Specify the type of angle in 'world coordinate system' or 'object coordinate system', which is no difference in Scannet dataset.")
    parser.add_argument("--use_normals", default=False, action="store_true")
    parser.add_argument("--hard_anchor", default=False, action="store_true")


    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=3, type=float)
    parser.add_argument("--matcher_center_cost", default=1, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)
    parser.add_argument("--matcher_size_cost", default=0.5, type=float)
    parser.add_argument("--matcher_anglecls_cost", default=0, type=float)
    parser.add_argument("--matcher_anglereg_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--cls_loss", default="focalloss_0.25", type=str)
    parser.add_argument("--loss_giou_weight", default=2, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=3, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=1, type=float)
    parser.add_argument("--loss_size_weight", default=0.5, type=float)
    parser.add_argument("--point_cls_loss_weight", default=0.05, type=float)
    
    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"], default="scannet"
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=8, type=int)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument("--filt_empty", default=True, action="store_true")
    
    parser.add_argument("--rot_ratio", default=5.0, type=float)
    parser.add_argument("--trans_ratio", default=0.4, type=float)
    parser.add_argument("--scale_ratio", default=0.4, type=float)
    parser.add_argument("--normal_trans", default=False, action="store_true")
    
    parser.add_argument("--use_color", default=False, action="store_true") 
    parser.add_argument("--xyz_color", default=False, action="store_true") 
    parser.add_argument("--color_drop", default=0.0, type=float)
    parser.add_argument("--color_contrastp", default=0.0, type=float)
    parser.add_argument("--color_jitterp", default=0.0, type=float)
    parser.add_argument("--hue_sat", default="0.5_0.2_0.0", type=str)
    parser.add_argument("--color_mean", default=-1, type=float)
    parser.add_argument("--coloraug_sunrgbd", default=False, action="store_true")
    
    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=540, type=int)
    parser.add_argument("--step_epoch", default="", type=str)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--auto_test", default=False, action="store_true")
    parser.add_argument("--test_no_nms", default=False, action="store_true",help="if you want use the model without NMS, please set the repeat_num==0")
    parser.add_argument("--no_3d_nms", default=False, action="store_true")
    parser.add_argument("--rotated_nms", default=False, action="store_true")
    parser.add_argument("--nms_iou", default=0.25, type=float)
    parser.add_argument("--empty_pt_thre", default=5, type=int)
    parser.add_argument("--conf_thresh", default=0.0, type=float)
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--angle_nms", default=False, action="store_true")
    parser.add_argument("--angle_conf", default=False, action="store_true")
    parser.add_argument("--use_old_type_nms", default=False, action="store_true")
    parser.add_argument("--no_cls_nms", default=False, action="store_true")
    parser.add_argument("--no_per_class_proposal", default=False, action="store_true")
    parser.add_argument("--use_cls_confidence_only", default=False, action="store_true")
    parser.add_argument("--test_size", default=False, action="store_true")

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=1, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    ##### wandb settings #####
    parser.add_argument("--wandb_activate", default=True, type=bool)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project", default="vdetr", type=str)
    parser.add_argument("--wandb_key", default="", type=str)

    return parser

def auto_reload(args):
    ignore_keys = [
    "test_only", "auto_test", "test_no_nms", "no_3d_nms", "rotated_nms",
    "nms_iou", "empty_pt_thre", "conf_thresh", "test_ckpt", "angle_nms",
    "angle_conf", "use_old_type_nms", "no_cls_nms", "filt_empty",
    "no_per_class_proposal", "use_cls_confidence_only", "test_size",
    "ngpus","dist_url","model_name","dataset_root_dir","meta_data_dir","checkpoint_dir",
    ]

    ckpt = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    ckpt_args = ckpt["args"]
    for arg_name in vars(ckpt_args):
        if arg_name not in ignore_keys and hasattr(args, arg_name):
            if getattr(args, arg_name) != getattr(ckpt_args, arg_name):
                print(arg_name,getattr(args, arg_name),getattr(ckpt_args, arg_name))
                setattr(args, arg_name, getattr(ckpt_args, arg_name))



def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)
            
        aps, wandb_iter, wandb_lr, wandb_loss, wandb_loss_details = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        curr_iter = epoch * len(dataloaders["train"])
        use_evaluate = ((epoch != 0) and (epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1))) or (epoch == 10)

        if is_primary():
            log_message = dict(\
                lr=wandb_lr,
                loss=wandb_loss,
                loss_cls=wandb_loss_details['loss_sem_cls'].item(),
                loss_angle_cls=wandb_loss_details['loss_angle_cls'].item(),
                loss_angle_reg=wandb_loss_details['loss_angle_reg'].item(),
                loss_center=wandb_loss_details['loss_center'].item() if 'loss_center' in wandb_loss_details else 0.0,
                loss_size=wandb_loss_details['loss_size'].item() if 'loss_size' in wandb_loss_details else 0.0,
                loss_giou=wandb_loss_details['loss_giou'].item() if 'loss_giou' in wandb_loss_details else 0.0,
                )
            if 'enc_point_cls_loss' in wandb_loss_details:
                log_message_enc=dict(\
                    enc_point_cls_loss=wandb_loss_details['enc_point_cls_loss'].item(),
                    )
                log_message.update(log_message_enc)#enc_point_cls_loss

            if args.wandb_activate:
                wandb_log(
                    data=log_message,
                    step=wandb_iter,
                    commit=not use_evaluate
                )

        if (
            epoch > args.max_epoch * 0.90
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ) or (epoch > args.max_epoch * 0.6 and epoch % args.eval_every_epoch == 0):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

        if use_evaluate:
            if epoch > args.max_epoch * 0.6:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                )
            ap_calculator, wandb_val_loss, wandb_val_loss_details = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                curr_iter,
            )
            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)

                log_message = dict(\
                    val_loss=wandb_val_loss,
                    val_AP25=metrics_dict['mAP_0.25'],
                    val_AP50=metrics_dict['mAP_0.5'],
                    val_AR25=metrics_dict['AR_0.25'],
                    val_AR50=metrics_dict['AR_0.5'],
                    val_loss_cls=wandb_val_loss_details['loss_sem_cls'].item(),
                    val_loss_angle_cls=wandb_val_loss_details['loss_angle_cls'].item(),
                    val_loss_angle_reg=wandb_val_loss_details['loss_angle_reg'].item(),
                    val_loss_center=wandb_val_loss_details['loss_center'].item(),
                    val_loss_size=wandb_val_loss_details['loss_size'].item(),               
                    val_loss_giou=wandb_val_loss_details['loss_giou'].item() if 'loss_giou' in wandb_val_loss_details else 0.0,) 
                if 'enc_point_cls_loss' in wandb_val_loss_details:
                    log_message_enc=dict(\
                        val_enc_point_cls_loss=wandb_val_loss_details['enc_point_cls_loss'].item(),
                        )
                    log_message.update(log_message_enc)
                if args.wandb_activate:
                    wandb_log(
                        data=log_message,
                        step=wandb_iter,
                    )
                
            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )

    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator, wandb_val_loss, wandb_val_loss_details = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)


def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"],strict=False)
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator, _ , _ = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)
    if args.test_size:
        metrics = ap_calculator.compute_metrics(size='S')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model S; Metrics {metric_str}")
            print("==" * 10)
        metrics = ap_calculator.compute_metrics(size='M')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model M; Metrics {metric_str}")
            print("==" * 10)
        metrics = ap_calculator.compute_metrics(size='L')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model L; Metrics {metric_str}")
            print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())
    if args.test_only and args.auto_test:
        auto_reload(args)

    datasets, dataset_config = build_dataset(args)
    model = build_model(args, dataset_config)
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        if args.mink_syncbn:
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],find_unused_parameters = True,
        )
    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])
        
        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=datasets[split].collate_fn
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_primary():
            # setup wandb
            if args.wandb_activate:
                wandb.login(key=args.wandb_key)
                run = wandb.init(
                    id=args.checkpoint_dir.split('/')[-1],
                    name=args.checkpoint_dir.split('/')[-1],
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    config=args,
                )



        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
