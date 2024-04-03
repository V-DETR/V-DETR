# Copyright (c) V-DETR authors. All Rights Reserved.

from typing import Optional
from functools import partial

import copy
import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.ops import points_in_boxes_all
                              
from models.position_embedding import PositionEmbeddingCoordsSine           
from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, 
                            GenericMLP, get_clones, PositionEmbeddingLearned)

from utils.pc_util import scale_points, shift_scale_points
class BoxProcessor(object):
    """
    Class to convert V-DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config, cls_loss="celoss"):
        self.dataset_config = dataset_config
        self.cls_loss = cls_loss

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_class_size(self, size_normalized_offset, logits):
        class_idx = logits.sigmoid().max(dim=-1)[1]
        size_per_class = torch.tensor(self.dataset_config.mean_size_arr, device=logits.device).float()
        class_size = size_per_class[class_idx]
        return class_size + size_normalized_offset*class_size, size_normalized_offset*class_size
    
    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual, zero_angle=False):
        if angle_logits.shape[-1] == 1 or zero_angle:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            if angle_logits.shape[-1] == 1:
                angle = angle_logits * 0 + angle_residual * 0
                angle = angle.squeeze(-1).clamp(min=0)
            else:
                angle = angle_logits.sum(-1) * 0 + angle_residual.sum(-1) * 0
                angle = angle.squeeze(-1).clamp(min=0)
            angle_prob = angle
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            angle_prob = torch.nn.functional.softmax(angle_logits, dim=-1)
            angle_prob, pred_angle_class = angle_prob.max(dim=-1)
            pred_angle_class = pred_angle_class.detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle, angle_prob

    def compute_objectness_and_cls_prob(self, cls_logits):
        if self.cls_loss.split('_')[0] == "focalloss":
            # assert cls_logits.shape[-1] == self.dataset_config.num_semcls
            cls_prob = cls_logits
            objectness_prob = cls_prob.sigmoid().max(dim=-1)[0]
            return cls_prob, objectness_prob
        else:
            assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
            cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
            objectness_prob = 1 - cls_prob[..., -1]
            return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def convert_corners_camera2lidar(corners_camera):
    corners_lidar = corners_camera
    corners_lidar[..., 1] *= -1 # X, -Z, Y
    corners_lidar[..., [0, 1, 2]] = corners_lidar[..., [0, 2, 1]]
    return corners_lidar


class TransformerDecoder(nn.Module):

    def __init__(self, first_layer, decoder_layer, dataset_config, 
                num_layers, decoder_dim=256, 
                mlp_dropout=0.3,
                mlp_norm="bn1d",
                mlp_act="relu",
                mlp_sep=False,
                pos_for_key=False,
                num_queries=256,
                cls_loss="celoss",
                norm_fn_name="ln",
                is_bilable=False,
                q_content='sample',
                return_intermediate=False,
                weight_init_name="xavier_uniform",
                args=None):
        super().__init__()
        self.first_layer = first_layer
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.dec_output_dim = self.layers[0].linear2.out_features
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.dec_output_dim)
        self.is_bilable = is_bilable
         
        self.pos_for_key = pos_for_key
        self.num_queries = num_queries
        self.q_content = q_content


        self.query_pos_projection = nn.ModuleList()
        for _ in range(self.num_layers):
            self.query_pos_projection.append(
                PositionEmbeddingLearned(6, self.dec_output_dim))
        if self.pos_for_key:
            self.key_pos_projection = nn.ModuleList()
            for _ in range(self.num_layers):
                self.key_pos_projection.append(
                    PositionEmbeddingLearned(3, self.dec_output_dim))

        if self.q_content == 'random' or self.q_content == 'random_add':
            self.query_embed = nn.Embedding(self.num_queries, self.dec_output_dim)
        
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

        self.mlp_norm = mlp_norm
        self.mlp_act = mlp_act
        self.mlp_sep = mlp_sep
        self.cls_loss = cls_loss

        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)
        self.build_pointcls_heads(dataset_config, decoder_dim, mlp_dropout)
        
        if self.cls_loss.split('_')[0] == "focalloss":
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            num_cls_first_layer = 1 if self.is_bilable else dataset_config.num_semcls
            self.mlp_heads[0]["sem_cls_head"].layers[-1].bias.data = torch.ones(num_cls_first_layer) * bias_value
            for i in range(1,num_layers+1):
                self.mlp_heads[i]["sem_cls_head"].layers[-1].bias.data = torch.ones(dataset_config.num_semcls) * bias_value

        for i in range(0, num_layers+1):
            nn.init.constant_(self.mlp_heads[i]["center_head"].layers[-1].weight.data, 0.0)
            nn.init.constant_(self.mlp_heads[i]["center_head"].layers[-1].bias.data, 0.0)
            nn.init.constant_(self.mlp_heads[i]["size_head"].layers[-1].weight.data, 0.0)
            nn.init.constant_(self.mlp_heads[i]["size_head"].layers[-1].bias.data, 0.0)
        self.box_processor = BoxProcessor(dataset_config, cls_loss=self.cls_loss)

    def build_pointcls_heads(self,dataset_config, decoder_dim, mlp_dropout):       
        mlp_func = partial(
            GenericMLP,
            norm_fn_name=self.mlp_norm,
            activation=self.mlp_act,
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )
        # Semantic class of the box
        # add 1 for background/not-an-object class
        if self.cls_loss.split('_')[0] == "focalloss":
            semcls_head = mlp_func(output_dim=dataset_config.num_semcls)
        else:
            semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)
        self.pointcls_heads = semcls_head 

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name=self.mlp_norm,
            activation=self.mlp_act,
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        if self.cls_loss.split('_')[0] == "focalloss":
            semcls_head = mlp_func(output_dim=dataset_config.num_semcls)
        else:
            semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        if self.mlp_sep:
            if self.is_bilable:
                self.mlp_heads = get_clones(nn.ModuleDict(mlp_heads), self.num_layers)
                first_heads = copy.deepcopy(mlp_heads)
                first_heads[0] = ("sem_cls_head", mlp_func(output_dim=1))
                self.mlp_heads.insert(0,nn.ModuleDict(first_heads))
            else:
                self.mlp_heads = get_clones(nn.ModuleDict(mlp_heads), self.num_layers+1)
        else:
            self.mlp_heads = nn.ModuleDict(mlp_heads)

    def _reset_parameters(self, weight_init_name):
        print('random init decoder')
        func = WEIGHT_INIT_DICT[weight_init_name]
        for name, p in self.named_parameters():
            if p.dim() > 1:
                func(p)
                
 
    def get_proposal_box_predictions_refine(self, idx, query_xyz, point_cloud_dims, box_features, pre_center_normalized=None,pre_size_normalized=None,):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(1,2,0)
        batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
        )
        # mlp head outputs are batch x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads[idx]["sem_cls_head"](box_features).transpose(1, 2)

        #prepare pre_size and pre_center
        scene_size = point_cloud_dims[1]-point_cloud_dims[0]
        pre_center_unnormalized = pre_center_normalized*scene_size.unsqueeze(1).repeat(1,num_queries,1)+point_cloud_dims[0].unsqueeze(1).repeat(1,num_queries,1)
        pre_size_unnormalized = pre_size_normalized*scene_size.unsqueeze(1).repeat(1,num_queries,1)

        #center
        assert  pre_center_normalized!=None
        center_reg =  self.mlp_heads[idx]["center_head"](box_features).transpose(1, 2).contiguous().view(batch,num_queries,3).contiguous()
        center_unnormalized = center_reg * pre_size_unnormalized + pre_center_unnormalized
        center_normalized = (center_unnormalized - point_cloud_dims[0].unsqueeze(1).repeat(1,num_queries,1))/scene_size.unsqueeze(1).repeat(1,num_queries,1)

            
        #size
        assert  pre_size_normalized!=None
        size_reg = self.mlp_heads[idx]["size_head"](box_features).transpose(1, 2).contiguous().view(batch,num_queries,3).contiguous()
        size_unnormalized = torch.exp(size_reg)*pre_size_unnormalized
        size_normalized = size_unnormalized/scene_size.unsqueeze(1).repeat(1,num_queries,1)

        #angle
        angle_logits = self.mlp_heads[idx]["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads[idx]["angle_residual_head"](
            box_features
        ).transpose(1, 2)
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )
        angle_continuous, angle_prob = self.box_processor.compute_predicted_angle(
            angle_logits, angle_residual
        )


        #conners
        box_corners = self.box_processor.box_parametrization_to_corners(
            center_unnormalized, size_unnormalized, angle_continuous
        )               
        angle_zero, _ = self.box_processor.compute_predicted_angle(
            angle_logits, angle_residual, zero_angle=True
        )
        box_corners_axis_align = self.box_processor.box_parametrization_to_corners(
            center_unnormalized, size_unnormalized, angle_zero
        )

        with torch.no_grad():
            (
                semcls_prob,
                objectness_prob,
            ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits)

        box_prediction = {
            "sem_cls_logits": cls_logits,
            "center_normalized": center_normalized.contiguous(),
            "center_unnormalized": center_unnormalized,
            "size_normalized": size_normalized,
            "size_unnormalized": size_unnormalized,
            "angle_logits": angle_logits,
            "angle_prob": angle_prob,
            "angle_residual": angle_residual,
            "angle_residual_normalized": angle_residual_normalized,
            "angle_continuous": angle_continuous,
            "objectness_prob": objectness_prob,
            "sem_cls_prob": semcls_prob,
            "box_corners": box_corners,
            "box_corners_axis_align": box_corners_axis_align,
        }
        #object-wise normalize
        box_prediction["pre_box_center_unnormalized"] = pre_center_unnormalized
        box_prediction["center_reg"] = center_reg
        box_prediction["pre_box_size_unnormalized"] = pre_size_unnormalized
        box_prediction["size_reg"] = size_reg    
        
        return box_prediction
           
    def forward(self, tgt, memory, query_xyz, enc_xyz, point_cloud_dims, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                enc_box_predictions: Optional[Tensor] = None,
                enc_box_features: Optional[Tensor] = None,
                ):
        
        intermediate = []
        attns = []

        mlp_idx = 0
        output= self.first_layer(enc_box_features)   
        attn = None    
        
        box_prediction = self.get_proposal_box_predictions_refine(
            mlp_idx, query_xyz, point_cloud_dims, self.norm(output),
            pre_center_normalized=enc_box_predictions["center_normalized"],
            pre_size_normalized=enc_box_predictions["size_normalized"],
        )    
    
        if self.return_intermediate:
            intermediate.append(box_prediction)

        dec_proposal_class = box_prediction["objectness_prob"].clone().detach()
        if dec_proposal_class.shape[1]>=self.num_queries:
            topk_proposals = torch.topk(dec_proposal_class, self.num_queries, dim=1)[1]
        else:
            topk_proposals = torch.arange(0,dec_proposal_class.shape[1],device=dec_proposal_class.device).unsqueeze(0).repeat(dec_proposal_class.shape[0],1)
        reference_point = convert_corners_camera2lidar(torch.gather(
                box_prediction["box_corners"].clone().detach(), 1, \
                topk_proposals.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)
            ))
        reference_center = torch.gather(
                box_prediction["center_unnormalized"].clone().detach(), 1, \
                topk_proposals.unsqueeze(-1).repeat(1, 1 ,3)
            )
        query_xyz = reference_center.clone().detach()
        reference_size = torch.gather(
                box_prediction["size_unnormalized"].clone().detach(), 1, \
                topk_proposals.unsqueeze(-1).repeat(1, 1 ,3)
            )
        reference_angle = torch.gather(
                box_prediction["angle_continuous"].clone().detach(), 1, \
                topk_proposals
            )
        # will used to refine in predition 
        proposal_center_normalized = torch.gather(
                box_prediction["center_normalized"].clone().detach(), 1, \
                topk_proposals.unsqueeze(-1).repeat(1, 1 ,3)
            )
        proposal_size_normalized = torch.gather(
                box_prediction["size_normalized"].clone().detach(), 1, \
                topk_proposals.unsqueeze(-1).repeat(1, 1 ,3)
            )
        output = torch.gather(
            output.permute(1,0,2), 1, \
            topk_proposals.unsqueeze(-1).repeat(1, 1, output.shape[-1])
            ).permute(1,0,2).contiguous()
        if self.q_content == 'zero':
            output = torch.zeros_like(output)
        elif self.q_content == 'random':
            output = self.query_embed.weight.unsqueeze(1).repeat(1, output.shape[1], 1)
        elif self.q_content == 'random_add':
            output = output + self.query_embed.weight.unsqueeze(1).repeat(1, output.shape[1], 1)                


        for idx, layer in enumerate(self.layers):
            if not idx == 0:
                reference_point = convert_corners_camera2lidar(box_prediction['box_corners'].clone().detach())
                reference_center = box_prediction['center_unnormalized'].clone().detach()
                reference_size = box_prediction['size_unnormalized'].clone().detach()
                reference_angle = box_prediction['angle_continuous'].clone().detach()

            query_reference = torch.cat([reference_center, reference_size], dim=-1)
            query_pos = self.query_pos_projection[idx](query_reference).permute(2, 0, 1)
            if self.pos_for_key:
                pos = self.key_pos_projection[idx](enc_xyz).permute(2, 0, 1)
            output, attn = layer(output, memory, 
                           reference_point, reference_angle, enc_xyz, point_cloud_dims,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_attn_weights=return_attn_weights)
            
            box_prediction = self.get_proposal_box_predictions_refine(
                idx+mlp_idx+1, query_xyz, point_cloud_dims, self.norm(output),
                pre_center_normalized=proposal_center_normalized, 
                pre_size_normalized=proposal_size_normalized,
            )    
        
            if self.return_intermediate:
                intermediate.append(box_prediction)
            if return_attn_weights:
                attns.append(attn)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            # bbox_predictions = torch.stack(intermediate)
            # intermediate decoder layer outputs are only used during training
            aux_outputs = intermediate[:-1]
            outputs = intermediate[-1]

            return {
                "outputs": outputs,  # output from last layer of decoder
                "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
            }, attns

        return {"outputs": box_prediction}, attns


class GlobalDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln", pos_for_key=False, args=None,):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.pos_for_key = pos_for_key
        
        if args.share_selfattn:
            self.self_attn = ShareSelfAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # cross attn layer
        self.multihead_attn = GlobalShareCrossAttention(d_model, nhead,
            attn_drop=dropout, proj_drop=dropout, args=args)
                
        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, reference_point, reference_angle, enc_xyz, point_cloud_dims,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                        key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.pos_for_key:
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                            key=self.with_pos_embed(memory, pos),
                            reference_point=reference_point, 
                            reference_angle=reference_angle,
                            xyz = enc_xyz,
                            attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)
        else:
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                            key=memory,
                            # If need pos for key
                            reference_point=reference_point, 
                            reference_angle=reference_angle,
                            xyz = enc_xyz,
                            attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory, reference_point, reference_angle, enc_xyz, point_cloud_dims,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                        key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.pos_for_key:
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                            key=self.with_pos_embed(memory, pos),
                            reference_point=reference_point, 
                            reference_angle=reference_angle,
                            xyz = enc_xyz,
                            attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)
        else:
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                            key=memory,
                            # If need pos for key
                            reference_point=reference_point, 
                            reference_angle=reference_angle,
                            xyz = enc_xyz,
                            attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory, reference_point, reference_angle, enc_xyz, point_cloud_dims,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, reference_point, reference_angle, enc_xyz, point_cloud_dims, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, reference_point, reference_angle, enc_xyz, point_cloud_dims, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)


class FFNLayer(nn.Module):
    def __init__(self, d_model,  dim_feedforward=256,
                 dropout=0.1,norm_fn_name="ln",
                 activation="relu", normalize_before=True,):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = NORM_DICT[norm_fn_name](d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        
    def forward_pre(self, memory,):
        memory = self.norm(memory)
        memory2 = self.linear2(self.dropout(self.activation(self.linear1(memory))))
        memory = memory + self.dropout(memory2)
        return memory

    def forward(self,memory,):
        return self.forward_pre(memory)


class ShareSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        dropout=0.0,
        args=None
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value=None, attn_mask=None, key_padding_mask=None):
        assert attn_mask == None and key_padding_mask == None
        key, query = key.permute(1,0,2), query.permute(1,0,2)
                                    
        B_, N, C = key.shape
        k = self.k(key).reshape(B_, N, 1, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B_, N, 1, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x).permute(1,0,2)
        return x, None
    

class GlobalShareCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        args=None
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.log_scale = args.log_scale
        self.rpe_quant = args.rpe_quant
        self.angle_type = args.angle_type

        self.interp_method, max_value, num_points = self.rpe_quant.split('_')
        max_value, num_points = float(max_value), int(num_points)
        relative_coords_table = torch.stack(torch.meshgrid(
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
        ), dim=-1).unsqueeze(0)
        self.register_buffer("relative_coords_table", relative_coords_table)
        self.max_value = max_value
        self.cpb_mlps = get_clones(self.build_cpb_mlp(3, args.rpe_dim, num_heads), 8)
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def forward(self, query, key, reference_point, reference_angle, xyz, attn_mask=None, key_padding_mask=None):
        # query: nQ, B, Dim (256)
        # key: nP, B, Dim
        # refer: B, nQ, 8, 3
        # xyz: B, nP, 3
        key, query = key.permute(1,0,2), query.permute(1,0,2)

        B, nQ = reference_point.shape[:2]
        nK = xyz.shape[1]
        for i in range(8):
            deltas = reference_point[:,:,None,i,:] - xyz[:,None,:,:]
            if self.angle_type == "object_coords" and reference_angle is not None:
                deltas[..., 2] *= -1 
                deltas[..., [0, 1, 2]] = deltas[..., [0, 2, 1]] # X,Y,Z -> X, -Z, Y

                R = roty_batch_tensor(reference_angle) # 4, 256, 3, 3
                deltas = torch.matmul(deltas, R)

                deltas[..., 1] *= -1 
                deltas[..., [0, 1, 2]] = deltas[..., [0, 2, 1]] # X, -Z, Y -> X,Y,Z

            deltas = torch.sign(deltas) * torch.log2(torch.abs(deltas)*self.log_scale + 1.0) / np.log2(8)
            delta = deltas / self.max_value # B, nQ, nP, 3
            
            rpe_table = self.cpb_mlps[i](self.relative_coords_table).permute(0, 4, 1, 2, 3) # B, 10, 10, 10, nH
            if i == 0:
                rpe = F.grid_sample(rpe_table, delta.view(1, 1, 1, -1, 3).to(rpe_table.dtype), mode=self.interp_method) \
                        .squeeze().view(-1, B, nQ, nK).permute(1, 0, 2, 3)
            else:
                rpe += F.grid_sample(rpe_table, delta.view(1, 1, 1, -1, 3).to(rpe_table.dtype), mode=self.interp_method) \
                        .squeeze().view(-1, B, nQ, nK).permute(1, 0, 2, 3)
                  
        B_, N_key, C = key.shape
        k = self.k(key).reshape(B_, N_key, 1, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key).reshape(B_, N_key, 1, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N_query, C = query.shape
        q = self.q(query).reshape(B_, N_query, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # 3dv_rpe to influnce attn
        attn += rpe
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            #attn_mask = attn_mask.reshape(attn.shape)
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float(-100))
            else:
                attn += attn_mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N_query, C)
        x = self.proj(x)
        x = self.proj_drop(x).permute(1,0,2)
        return x, attn


def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(
        tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device
    )
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output

def rotz_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(
        tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device
    )
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 1] = -s
    output[..., 1, 0] = s
    output[..., 1, 1] = c
    output[..., 2, 2] = 1
    return output