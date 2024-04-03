# ------------------------------------------------------------------------
# V-DETR
# Copyright (c) V-DETR authors. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from :
# 3DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Group-Free-3D
# Copyright (c) Group-Free-3D authors. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment
from mmcv.ops import points_in_boxes_all
from mmcv.ops import diff_iou_rotated_3d
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d
    

def diff_diou_rotated_3d(box3d1, box3d2):
    """Calculate differentiable iou of rotated 3d boxes.
    Args:
        box3d1 (Tensor): (B, N, 3+3+1) First box (x,y,z,w,h,l,alpha).
        box3d2 (Tensor): (B, N, 3+3+1) Second box (x,y,z,w,h,l,alpha).
    Returns:
        Tensor: (B, N) IoU.
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) -
                 torch.max(zmin1, zmin2)).clamp_(min=0.)
    intersection_3d = intersection * z_overlap
    volume1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    volume2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    union_3d = volume1 + volume2 - intersection_3d

    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    z_max = torch.max(zmax1, zmax2)
    z_min = torch.min(zmin1, zmin2)

    r2 = ((box1[..., :3] - box2[..., :3]) ** 2).sum(dim=-1)
    c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2
    
    return intersection_3d / union_3d - r2 / c2


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class Matcher(nn.Module):
    def __init__(self, cls_loss, cost_class, cost_objectness, cost_giou, cost_center, cost_size, args):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cls_loss = cls_loss
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center
        self.cost_size = cost_size
        self.matcher_anglecls_cost = args.matcher_anglecls_cost
        self.matcher_anglereg_cost = args.matcher_anglereg_cost

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        if self.cls_loss.split('_')[0] == "focalloss":
            pred_cls_prob = outputs["sem_cls_prob"].sigmoid()
            gt_box_sem_cls_labels = (
                targets["gt_box_sem_cls_label"]
                .unsqueeze(1)
                .expand(batchsize, nqueries, ngt)
            )

            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (pred_cls_prob ** gamma) * (-(1 - pred_cls_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - pred_cls_prob) ** gamma) * (-(pred_cls_prob + 1e-8).log())
            class_mat = torch.gather(pos_cost_class-neg_cost_class, 2, gt_box_sem_cls_labels)
        else:
            pred_cls_prob = outputs["sem_cls_prob"]
            gt_box_sem_cls_labels = (
                targets["gt_box_sem_cls_label"]
                .unsqueeze(1)
                .expand(batchsize, nqueries, ngt)
            )
            class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        pred_anglecls_prob = outputs["angle_logits"]
        gt_box_angle_cls_labels = (
                targets["gt_angle_class_label"]
                .unsqueeze(1)
                .expand(batchsize, nqueries, ngt)
            )
        angle_class_mat = -torch.gather(pred_anglecls_prob, 2, gt_box_angle_cls_labels).detach()

        angle_residual = outputs["angle_residual_normalized"]
        gt_angle_label = targets["gt_angle_class_label"]
        gt_angle_residual = targets["gt_angle_residual_label"]
        gt_angle_residual_normalized = gt_angle_residual / (
            np.pi / angle_residual.shape[-1]
        )
        gt_angle_label_one_hot = torch.zeros_like(
            angle_residual, dtype=torch.float32
        ).unsqueeze(2).repeat(1, 1, gt_angle_label.shape[1], 1)
        # import pdb; pdb.set_trace()
        gt_angle_label = gt_angle_label.unsqueeze(1).repeat(1, angle_residual.shape[1], 1)
        gt_angle_label_one_hot.scatter_(3, gt_angle_label.unsqueeze(-1), 1)
        angle_residual = angle_residual.unsqueeze(2).repeat(1, 1, gt_angle_label.shape[2], 1)

        angle_residual_for_gt_class = torch.sum(
            angle_residual * gt_angle_label_one_hot, -1
        )
       
        angle_reg_mat = huber_loss(
            angle_residual_for_gt_class - gt_angle_residual_normalized.unsqueeze(1), delta=1.0
        ).detach()
            
        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
 
        center_mat = outputs["center_reg_dist"].detach()
        size_mat = outputs["size_reg_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
           self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
            + self.cost_size * size_mat
            + self.matcher_anglecls_cost * angle_class_mat
            + self.matcher_anglereg_cost * angle_reg_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, args, matcher, dataset_config, loss_weight_dict, ):
        super().__init__()
        self.args = args
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict
        self.is_bilable = args.is_bilable
        self.repeat_num = args.repeat_num
        self.iou_type = args.iou_type
    
        if self.args.cls_loss.split('_')[0] == "focalloss":
            self.focal_alpha = float(self.args.cls_loss.split('_')[1])
            del loss_weight_dict["loss_no_object_weight"]
        else:
            semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
            semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
            del loss_weight_dict["loss_no_object_weight"]
            self.register_buffer("semcls_percls_weights", semcls_percls_weights)
    
        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}
    
    def loss_point_cls(self, outputs, targets):
        if targets["num_boxes_replica"] > 0:
            target_boxes = torch.cat((targets["gt_box_centers"], targets["gt_box_sizes"], targets["gt_box_angles"].unsqueeze(-1)),dim=-1)
            # cx, cy, cz, w, h, l, 0
            target_boxes[...,:,2] = target_boxes[...,:,2] - target_boxes[...,:,5] / 2  # bottom center
            seed_xyz = outputs['seed_xyz']
            inbox_inds = points_in_boxes_all(seed_xyz,target_boxes) # bs, npoints, nboxes
            # select only one box if point in different boxes#size+argmin
            bs, npoints, nboxes = inbox_inds.shape
            for i in range(bs):
                inbox_inds[i, ...,targets['nactual_gt'][i]:] = 0 # mask
            gt_box_volume = targets['gt_box_sizes'][...,0]*targets['gt_box_sizes'][...,1]*targets['gt_box_sizes'][...,2]
            inbox_inds_with_volume = inbox_inds * gt_box_volume.unsqueeze(1).repeat(1,npoints,1)
            inbox_inds_with_volume[inbox_inds_with_volume==0] = 1000 #invalid size
            
            inbox_inds_with_volume = torch.cat((inbox_inds_with_volume, inbox_inds_with_volume.new_ones(bs,npoints,1)*100),dim=-1)
            point_assignment = torch.argmin(inbox_inds_with_volume,dim=-1)
            point_assignment_mask = (point_assignment!=nboxes).int()
            point_assignment[point_assignment==nboxes] = 0
            
            
            assignments= {
                "per_prop_gt_inds": point_assignment,#[bs,n_query] each query to gt
                "proposal_matched_mask": point_assignment_mask,#[bs,n_query] whether query is matched with one gt
            }        

            if self.args.cls_loss.split('_')[0] == "focalloss":
                pred_logits = outputs["point_cls_logits"]
                gt_box_label = torch.gather(
                    targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
                )
                gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
                    pred_logits.shape[-1]
                )
                target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                                    dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
                target_classes_onehot.scatter_(2, gt_box_label.unsqueeze(-1), 1)

                target_classes_onehot = target_classes_onehot[:,:,:-1]
                loss = sigmoid_focal_loss(pred_logits, target_classes_onehot, \
                    targets["num_boxes"], alpha=self.focal_alpha, gamma=2) * pred_logits.shape[1]
            else:
                pred_logits = outputs["point_cls_logits"]
                gt_box_label = torch.gather(
                    targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
                )
                gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
                    pred_logits.shape[-1] - 1
                )
                
                loss = F.cross_entropy(
                    pred_logits.transpose(2, 1),
                    gt_box_label,
                    self.semcls_percls_weights,
                    reduction="mean",
                )
        else:
            loss = outputs["point_cls_logits"].sum() * 0.0

        return loss

    def loss_sem_cls(self, outputs, targets, assignments):

        if targets["num_boxes_replica"] > 0:
            if self.args.cls_loss.split('_')[0] == "focalloss":
                pred_logits = outputs["sem_cls_logits"]
                gt_box_label = torch.gather(
                    targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
                )
                gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
                    pred_logits.shape[-1]
                )

                target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                                    dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
                target_classes_onehot.scatter_(2, gt_box_label.unsqueeze(-1), 1)

                target_classes_onehot = target_classes_onehot[:,:,:-1]
                loss = sigmoid_focal_loss(pred_logits, target_classes_onehot, \
                    targets["num_boxes"], alpha=self.focal_alpha, gamma=2) * pred_logits.shape[1]
                
                # loss = F.cross_entropy(
                #     pred_logits.transpose(2, 1),
                #     gt_box_label,
                #     self.semcls_percls_weights,
                #     reduction="mean",
                # )
            else:
                pred_logits = outputs["sem_cls_logits"]
                gt_box_label = torch.gather(
                    targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
                )
                gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
                    pred_logits.shape[-1] - 1
                )
                loss = F.cross_entropy(
                    pred_logits.transpose(2, 1),
                    gt_box_label,
                    self.semcls_percls_weights,
                    reduction="mean",
                )
        else:
            loss = outputs["sem_cls_logits"].sum() * 0.0

        return {"loss_sem_cls": loss}

    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )
            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            
            # angle_reg_loss2 = torch.gather(
            #     assignments["angle_reg_mat"], 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            # ).squeeze(-1)
            # import pdb; pdb.set_trace()
            
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = outputs["angle_logits"].sum() * 0.0
            angle_reg_loss = outputs["angle_residual_normalized"].sum() * 0.0
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):

        center_dist = outputs["center_reg_dist"]

        if targets["num_boxes_replica"] > 0:
            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = outputs['center_reg'].sum() * 0.0 + outputs['pre_box_center_unnormalized'].sum() * 0.0 + \
                outputs['pre_box_size_unnormalized'].sum() * 0.0

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]

        if targets["num_boxes_replica"] > 0:
            # # Non vectorized version
            # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
            # assign = assignments["assignments"]

            # for b in range(gious_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate gious by using proposal to gt matching 
            giou_loss = torch.gather(
                gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            giou_loss = giou_loss * assignments["proposal_matched_mask"]
            giou_loss = giou_loss.sum()

            if targets["num_boxes"] > 0:
                giou_loss /= targets["num_boxes"]
        else:
            giou_loss = outputs["size_normalized"].sum() * 0.0 + outputs["center_normalized"].sum() * 0.0

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):

        gt_box_sizes = targets["gt_box_sizes"]
        pred_box_sizes = outputs["size_unnormalized"]


        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            
            gt_size_reg = torch.log((gt_box_sizes + 1e-5)/(outputs["pre_box_size_unnormalized"] + 1e-5)) #bs,prop,3
            pred_size_reg = outputs["size_reg"]#bs,prop,3
            size_loss = F.l1_loss(gt_size_reg, pred_size_reg, reduction="none").sum(
                dim=-1
            )        

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = outputs["pre_box_size_unnormalized"].sum() * 0.0 + outputs["size_reg"].sum() * 0.0

        return {"loss_size": size_loss}

    def repeat_ground_truth(self, maingt_repeat, batch_data_label):
        batch_data_label_multi = copy.deepcopy(batch_data_label)
        batch_data_label_multi["gt_box_corners"] = batch_data_label_multi[
            "gt_box_corners"
        ].repeat(1, maingt_repeat, 1, 1)
        batch_data_label_multi["gt_box_centers"] = batch_data_label_multi[
            "gt_box_centers"
        ].repeat(1, maingt_repeat, 1)
        batch_data_label_multi["gt_box_centers_normalized"] = batch_data_label_multi[
            "gt_box_centers_normalized"
        ].repeat(1, maingt_repeat, 1)

        batch_data_label_multi["gt_box_sem_cls_label"] = batch_data_label_multi[
            "gt_box_sem_cls_label"
        ].repeat(1, maingt_repeat)
        batch_data_label_multi["gt_box_present"] = batch_data_label_multi[
            "gt_box_present"
        ].repeat(1, maingt_repeat)

        batch_data_label_multi["gt_box_sizes"] = batch_data_label_multi[
            "gt_box_sizes"
        ].repeat(1, maingt_repeat, 1)
        batch_data_label_multi["gt_box_sizes_normalized"] = batch_data_label_multi[
            "gt_box_sizes_normalized"
        ].repeat(1, maingt_repeat, 1)

        batch_data_label_multi["gt_box_angles"] = batch_data_label_multi[
            "gt_box_angles"
        ].repeat(1, maingt_repeat)
        batch_data_label_multi["gt_angle_class_label"] = batch_data_label_multi[
            "gt_angle_class_label"
        ].repeat(1, maingt_repeat)
        batch_data_label_multi["gt_angle_residual_label"] = batch_data_label_multi[
            "gt_angle_residual_label"
        ].repeat(1, maingt_repeat)

        batch_size = batch_data_label_multi["scan_idx"].shape[0]
        for b in range(batch_size):
            valid_mask = batch_data_label_multi["gt_box_present"][b] > 0
            valid_num = valid_mask.sum()
            batch_data_label_multi["gt_box_corners"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_corners"][b][valid_mask]
            batch_data_label_multi["gt_box_centers"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_centers"][b][valid_mask]
            batch_data_label_multi["gt_box_centers_normalized"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_centers_normalized"][b][valid_mask]

            batch_data_label_multi["gt_box_sem_cls_label"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_sem_cls_label"][b][valid_mask]
            batch_data_label_multi["gt_box_present"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_present"][b][valid_mask]

            batch_data_label_multi["gt_box_sizes"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_sizes"][b][valid_mask]
            batch_data_label_multi["gt_box_sizes_normalized"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_sizes_normalized"][b][valid_mask]

            batch_data_label_multi["gt_box_angles"][b][
                :valid_num
            ] = batch_data_label_multi["gt_box_angles"][b][valid_mask]
            batch_data_label_multi["gt_angle_class_label"][b][
                :valid_num
            ] = batch_data_label_multi["gt_angle_class_label"][b][valid_mask]
            batch_data_label_multi["gt_angle_residual_label"][b][
                :valid_num
            ] = batch_data_label_multi["gt_angle_residual_label"][b][valid_mask]
        
            for k in batch_data_label_multi.keys():
                if k not in  ["nactual_gt","num_boxes","num_boxes_replica",'scan_idx']:
                    batch_data_label_multi[k][b][valid_num:] = 0

        nactual_gt = batch_data_label_multi["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        batch_data_label_multi["nactual_gt"] = nactual_gt
        batch_data_label_multi["num_boxes"] = num_boxes
        batch_data_label_multi[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        return batch_data_label_multi
  
    def single_output_forward(self, outputs, targets):
        if self.iou_type == 'diou' or self.iou_type == 'iou':
            gt_bbox = torch.cat([targets["gt_box_centers"], targets["gt_box_sizes"], targets["gt_box_angles"].unsqueeze(-1)], dim=-1)
            pred_bbox = torch.cat([outputs["center_unnormalized"], outputs["size_unnormalized"], outputs["angle_continuous"].unsqueeze(-1)], dim=-1)
            gt_num, pred_num = gt_bbox.shape[1], pred_bbox.shape[1]
            gt_bbox = gt_bbox.unsqueeze(1).repeat(1, pred_num, 1, 1)
            pred_bbox = pred_bbox.unsqueeze(2).repeat(1, 1, gt_num, 1)
            gious = diff_diou_rotated_3d(pred_bbox.flatten(1,2), gt_bbox.flatten(1,2)) if self.iou_type == 'diou' \
                else diff_iou_rotated_3d(pred_bbox.flatten(1,2), gt_bbox.flatten(1,2))
            gious = gious.reshape(-1, pred_num, gt_num)
            mask = torch.zeros(gious.shape, device=gious.device, dtype=torch.float32)
            for b in range(gious.shape[0]):
                mask[b, :, : targets["nactual_gt"][b]] = 1
            gious *= mask
        else:
            gious = generalized_box3d_iou(outputs["box_corners"],targets["gt_box_corners"],targets["nactual_gt"],rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),)

        outputs["gious"] = gious
        

        gt_center_reg = (targets['gt_box_centers'].unsqueeze(1) - outputs['pre_box_center_unnormalized'].unsqueeze(2)) / (outputs['pre_box_size_unnormalized'].unsqueeze(2) + 1e-5)
        outputs["center_reg_dist"] =  torch.abs(outputs['center_reg'].unsqueeze(2) - gt_center_reg).sum(-1) # L1dist: bs,npred,ngt

        gt_size_reg = torch.log((targets['gt_box_sizes'].unsqueeze(1) + 1e-5) / (outputs['pre_box_size_unnormalized'].unsqueeze(2) + 1e-5))
        outputs["size_reg_dist"] =  torch.abs(outputs['size_reg'].unsqueeze(2) - gt_size_reg).sum(-1) # L1dist:bs,npred,ngt


        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        if self.repeat_num > 1:
            targets_repeated = self.repeat_ground_truth(self.repeat_num, targets)
            loss, loss_dict = self.single_output_forward(outputs["outputs"], targets_repeated)     
        else:
            loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                if k == 0 and self.is_bilable: #only to distinguish positive and negative
                    bin_targets = copy.deepcopy(targets)
                    bin_targets['gt_box_sem_cls_label'] = torch.zeros_like(targets['gt_box_sem_cls_label'])
                    interm_loss, interm_loss_dict = self.single_output_forward(
                        outputs["aux_outputs"][k], bin_targets,
                    )
                else: 
                    if self.repeat_num > 1:
                        targets_repeated = self.repeat_ground_truth(self.repeat_num, targets)
                        interm_loss, interm_loss_dict = self.single_output_forward(
                            outputs["aux_outputs"][k], targets_repeated,
                        )
                    else:
                        interm_loss, interm_loss_dict = self.single_output_forward(
                            outputs["aux_outputs"][k], targets,
                        )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]

        if "enc_outputs" in outputs:
            outputs["enc_outputs"]['seed_inds'] = outputs['seed_inds']
            outputs["enc_outputs"]['seed_xyz'] = outputs['seed_xyz'] 
            assert "point_cls_logits" in outputs["enc_outputs"]
            enc_point_cls_loss = self.loss_point_cls(outputs["enc_outputs"],targets)*self.args.point_cls_loss_weight
            loss += enc_point_cls_loss
            loss_dict["enc_point_cls_loss"] = enc_point_cls_loss
            
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cls_loss=args.cls_loss,
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
        cost_size=args.matcher_size_cost,
        args=args
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    criterion = SetCriterion(args, matcher, dataset_config, loss_weight_dict)
    return criterion
