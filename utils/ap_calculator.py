# ------------------------------------------------------------------------
# V-DETR
# Copyright (c) V-DETR authors. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from :
# 3DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import torch

from utils.box_util import (extract_pc_in_box3d, flip_axis_to_camera_np,
                            get_3d_box, get_3d_box_batch)
from utils.eval_det import eval_det, eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from mmcv.ops import nms3d, nms3d_normal, points_in_boxes_all
from functools import partial
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
import third_party.pointnet2.pointnet2_utils as pointnet2_utils


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


# This is exactly the same as VoteNet so that we can compare evaluations.
def parse_predictions(
    predicted_boxes, sem_cls_probs, objectness_probs, angle_probs, point_cloud, config_dict, predicted_boxes_CSA=None
):
    """Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
    pred_sem_cls = np.argmax(sem_cls_probs, -1)
    obj_prob = objectness_probs.detach().cpu().numpy()
    angle_probs = angle_probs.detach().cpu().numpy()

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy()

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        predicted_boxes_CSA[...,:,2] = predicted_boxes_CSA[...,:,2] - predicted_boxes_CSA[...,:,5]/2  #bottom center

        new_idx = torch.randperm(point_cloud.shape[1])[:40000]
        new_xyz = point_cloud[:,new_idx,:3]
        
        # sample_inds = pointnet2_utils.furthest_point_sample(point_cloud[:,:,0:3].contiguous(), 40000)
        # xyz_flipped = point_cloud[:,:,0:3].transpose(1, 2).contiguous()
        # new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()

        inbox_inds = points_in_boxes_all(new_xyz,predicted_boxes_CSA).permute(0,2,1).contiguous()#bs,nboxes,npoints
        num_points_inbox = inbox_inds.sum(-1)#bs,nboxes
        nonempty_box_mask = (num_points_inbox >= config_dict["empty_pt_thre"]).long().cpu().numpy()
        for i in range(bsize):
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1       

        predicted_boxes_CSA[...,:,2] = predicted_boxes_CSA[...,:,2] + predicted_boxes_CSA[...,:,5]/2  #box center
        # batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # B,N,3
        # for i in range(bsize):
        #     pc = batch_pc[i, :, :]  # (N,3)
        #     for j in range(K):
        #         box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
        #         box3d = flip_axis_to_depth(box3d)
        #         pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
        #         if len(pc_in_box) < config_dict["empty_pt_thre"]:
        #             nonempty_box_mask[i, j] = 0
        #     if nonempty_box_mask[i].sum() == 0:
        #         nonempty_box_mask[i, obj_prob[i].argmax()] = 1
        # -------------------------------------
    
    if config_dict["rotated_nms"]:
        print(pred_corners_3d_upright_camera[0, 0])
    elif "no_nms" in config_dict and config_dict["no_nms"]:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j] * angle_probs[i, j] if config_dict["angle_nms"] else obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = (
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["angle_conf"]:
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j] * angle_probs[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        elif config_dict["per_class_proposal"]:
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )

    return batch_pred_map_cls


def get_ap_config_dict(
    remove_empty_box=True,
    use_3d_nms=True,
    nms_iou=0.25,
    use_old_type_nms=False,
    cls_nms=True,
    per_class_proposal=True,
    use_cls_confidence_only=False,
    conf_thresh=0.0,
    no_nms=False,
    dataset_config=None,
    empty_pt_thre=5,
    rotated_nms=False,
    angle_nms=False,
    angle_conf=False
):
    """
    Default mAP evaluation settings for VoteNet
    """

    config_dict = {
        "remove_empty_box": remove_empty_box,
        "use_3d_nms": use_3d_nms,
        "nms_iou": nms_iou,
        "use_old_type_nms": use_old_type_nms,
        "cls_nms": cls_nms,
        "per_class_proposal": per_class_proposal,
        "use_cls_confidence_only": use_cls_confidence_only,
        "conf_thresh": conf_thresh,
        "no_nms": no_nms,
        "dataset_config": dataset_config,
        "empty_pt_thre": empty_pt_thre,
        "rotated_nms": rotated_nms,
        "angle_nms": angle_nms,
        "angle_conf": angle_conf,
    }
    return config_dict


class APCalculator(object):
    """Calculating Average Precision"""

    def __init__(
        self,
        dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=None,
        exact_eval=False,
        ap_config_dict=None,
        no_nms=False,
        args=None
    ):
        """
        Args:
            ap_iou_thresh: List of float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        if ap_config_dict is None:
            ap_config_dict = get_ap_config_dict(
                dataset_config=dataset_config, 
                remove_empty_box=exact_eval, 
                no_nms=no_nms, 
                use_3d_nms=not args.no_3d_nms,
                nms_iou=args.nms_iou, 
                empty_pt_thre=args.empty_pt_thre,
                conf_thresh=args.conf_thresh,
                rotated_nms=args.rotated_nms,
                angle_nms=args.angle_nms,
                angle_conf=args.angle_conf,
                use_old_type_nms=args.use_old_type_nms,
                cls_nms=not args.no_cls_nms,
                per_class_proposal=not args.no_per_class_proposal,
                use_cls_confidence_only=args.use_cls_confidence_only,
            )
        self.ap_config_dict = ap_config_dict
        self.class2type_map = class2type_map
        self.reset()

    def make_gt_list(self, gt_box_corners, gt_box_sem_cls_labels, gt_box_present):
        batch_gt_map_cls = []
        bsize = gt_box_corners.shape[0]
        for i in range(bsize):
            batch_gt_map_cls.append(
                [
                    (gt_box_sem_cls_labels[i, j].item(), gt_box_corners[i, j])
                    for j in range(gt_box_corners.shape[1])
                    if gt_box_present[i, j] == 1
                ]
            )
        return batch_gt_map_cls

    def step_meter(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]

        predicted_box_center = outputs['center_unnormalized'].detach()
        predicted_box_size = outputs['size_unnormalized'].detach()
        predicted_box_angle = outputs['angle_continuous'].detach()
        predicted_box_CSA = torch.cat((predicted_box_center,predicted_box_size,predicted_box_angle.unsqueeze(-1)),dim=-1)

        self.step(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            angle_probs=outputs["angle_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
            predicted_box_CSA=predicted_box_CSA,
        )

    def step(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        angle_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
        predicted_box_CSA,
    ):
        """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present
        )

        batch_pred_map_cls = parse_predictions(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            angle_probs,
            point_cloud,
            self.ap_config_dict,
            predicted_box_CSA
        )

        self.accumulate(batch_pred_map_cls, batch_gt_map_cls)

    def accumulate(self, batch_pred_map_cls, batch_gt_map_cls):
        """Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self, size=''):
        """Use accumulated predictions and groundtruths to compute Average Precision."""
        overall_ret = OrderedDict()
        for ap_iou_thresh in self.ap_iou_thresh:
            ret_dict = OrderedDict()
            rec, prec, ap = eval_det_multiprocessing(
                self.pred_map_cls, self.gt_map_cls, ovthresh=ap_iou_thresh, size=size,
            )
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                ret_dict["%s Average Precision" % (clsname)] = ap[key]
            ap_vals = np.array(list(ap.values()), dtype=np.float32)
            ap_vals[np.isnan(ap_vals)] = 0
            ret_dict["mAP"] = ap_vals.mean()
            rec_list = []
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                try:
                    ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                    rec_list.append(rec[key][-1])
                except:
                    ret_dict["%s Recall" % (clsname)] = 0
                    rec_list.append(0)
            ret_dict["AR"] = np.mean(rec_list)
            overall_ret[ap_iou_thresh] = ret_dict
        return overall_ret

    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)

    def metrics_to_str(self, overall_ret, per_class=True):
        mAP_strs = []
        AR_strs = []
        per_class_metrics = []
        for ap_iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
            mAP_strs.append(f"{mAP:.2f}")
            ar = overall_ret[ap_iou_thresh]["AR"] * 100
            AR_strs.append(f"{ar:.2f}")

            if per_class:
                # per-class metrics
                per_class_metrics.append("-" * 5)
                per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
                for x in list(overall_ret[ap_iou_thresh].keys()):
                    if x == "mAP" or x == "AR":
                        pass
                    else:
                        met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                        per_class_metrics.append(met_str)

        ap_header = [f"mAP{x:.2f}" for x in self.ap_iou_thresh]
        ap_str = ", ".join(ap_header)
        ap_str += ": " + ", ".join(mAP_strs)
        ap_str += "\n"

        ar_header = [f"AR{x:.2f}" for x in self.ap_iou_thresh]
        ap_str += ", ".join(ar_header)
        ap_str += ": " + ", ".join(AR_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ap_str += "\n"
            ap_str += per_class_metrics

        return ap_str

    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        for ap_iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{ap_iou_thresh}"] = (
                overall_ret[ap_iou_thresh]["mAP"] * 100
            )
            metrics_dict[f"AR_{ap_iou_thresh}"] = overall_ret[ap_iou_thresh]["AR"] * 100
        return metrics_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
