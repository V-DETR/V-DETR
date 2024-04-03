# Copyright (c) V-DETR authors. All Rights Reserved.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from third_party.pointnet2.pointnet2_utils import furthest_point_sample
import third_party.pointnet2.pointnet2_utils as pointnet2_utils
from models.mink_resnet import MinkResNet
from models.helpers import (GenericMLP, PositionEmbeddingLearned)
from models.position_embedding import PositionEmbeddingCoordsSine
from models.vdetr_transformer import (TransformerDecoder, GlobalDecoderLayer, FFNLayer)


class FPSModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xyz, features, num_proposal):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        # Farthest point sampling (FPS)
        sample_inds = pointnet2_utils.furthest_point_sample(xyz, num_proposal)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class ModelVDETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        num_queries=1024,
        querypos_mlp=False,
        minkowski=False,
        inplane=64,
        num_stages=4,
        voxel_size=0.01,
        npoint=2048,
        use_fpn=False,
        layer_idx=-1,
        proj_nohid=False,
        woexpand_conv=False,
        args=None
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.querypos_mlp = querypos_mlp
        self.minkowski = minkowski # other backbone will release soon
        self.voxel_size = voxel_size
        self.use_fpn = use_fpn
        self.layer_idx = layer_idx
        self.num_stages = num_stages
        self.proj_nohid = proj_nohid
        self.woexpand_conv = woexpand_conv
        self.npoint = npoint
        self.random_fps = args.random_fps
        self.use_color = args.use_color
        self.xyz_color = args.xyz_color
        
        if self.minkowski:
            self.fps_module = FPSModule()
            backbone_channels = [4 * inplane * 2**i for i in range(self.num_stages)] if args.depth > 34 \
                else [inplane * 2**i for i in range(self.num_stages)]
            
            self._init_fpn_layers(backbone_channels, encoder_dim)
        
        if self.encoder is not None:
            if hasattr(self.encoder, "masking_radius"):
                hidden_dims = [encoder_dim]
            else:
                hidden_dims = [encoder_dim, encoder_dim]
        else:
            if self.proj_nohid:
                hidden_dims = []
            else:
                hidden_dims = [encoder_dim]
            
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        if not self.querypos_mlp:
            self.pos_embedding = PositionEmbeddingCoordsSine(
                d_pos=decoder_dim, pos_type="fourier", normalize=True
            )
            self.query_projection = GenericMLP(
                input_dim=decoder_dim,
                hidden_dims=[decoder_dim],
                output_dim=decoder_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        self.decoder = decoder
        self.num_queries = num_queries

        self.dataset_config = dataset_config
        self.hard_anchor = args.hard_anchor

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _make_up_block(self, in_channels, out_channels, woexpand_conv):
        if woexpand_conv:
            return nn.Sequential(
                ME.MinkowskiConvolutionTranspose(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU()
            )
        else:
            return nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU()
            )

    def _init_fpn_layers(self, in_channels, out_channels):
        # neck layers
        # self.pruning = ME.MinkowskiPruning()
        if self.use_fpn:
            for i in range(self.layer_idx + 1, len(in_channels)):
                if i > 0:
                    self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1], self.woexpand_conv))
        self.__setattr__(f'out_block_{self.layer_idx}', self._make_block(in_channels[self.layer_idx], out_channels))
        
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            
    def get_query_embeddings(self, encoder_xyz, enc_features, point_cloud_dims):
        query_xyz = encoder_xyz
        query_inds = None

        if not self.querypos_mlp:
            pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
            query_embed = self.query_projection(pos_embed).permute(2, 0, 1)
        else:
            query_embed = query_xyz
        return query_xyz, query_embed, query_inds

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def single_scale_fps(self, out, num_sample):
        batch_num = (out.C[:,0]).max().int()+1
        features = out.F
        xyz = out.C[:,1:] * self.voxel_size

        sampled_features_batch = []
        sampled_xyz_batch = []
        for batch_id in range(batch_num):
            batch_id_list = out.C[:,0]
            batch_indices = torch.where(batch_id_list==batch_id)

            # select features and xyz
            features_batch = features[batch_indices]
            xyz_batch = xyz[batch_indices]

            # permute features  (Batch),Dims,Number
            features_batch = features_batch.transpose(1,0).contiguous()

            # sample to the same number
            ## Downsampling voxel points into specific number; (N=2048)
            xyz_batch_squ = xyz_batch.unsqueeze(0)
            features_batch_squ = features_batch.unsqueeze(0)

            sampled_xyz, sampled_features, sample_inds \
                = self.fps_module(xyz_batch_squ, features_batch_squ ,num_sample,)
                
            # store the sampled ones
            sampled_features_batch.append(sampled_features)
            sampled_xyz_batch.append(sampled_xyz)

        pre_enc_features = torch.cat(sampled_features_batch)
        pre_enc_xyz = torch.cat(sampled_xyz_batch)
        pre_enc_inds = None
        return pre_enc_features,pre_enc_xyz,pre_enc_inds

    def run_encoder(self, point_clouds,):

        if self.use_color:
            if self.xyz_color:
                coordinates, features = ME.utils.batch_sparse_collate(
                    [(p[:, :3] / self.voxel_size, p[:, :]) for p in point_clouds])
            else:
                coordinates, features = ME.utils.batch_sparse_collate(
                    [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in point_clouds])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, :3]) for p in xyz])
    
        origin_voxel = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.pre_encoder(origin_voxel)
        batch_num = origin_voxel.C[:,0].max().long() + 1
        inputs = x
            
        x = inputs[-1]  # Use the final layer output!
        for i in range(len(inputs) - 1, self.layer_idx - 1, -1):
            if self.use_fpn:
                if i < len(inputs) - 1:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    x = inputs[i] + x
                    # if self.prune:
                    #    x = self._prune(x, scores)
            else:
                x = inputs[i]
                
            if i == self.layer_idx:
                out = self.__getattr__(f'out_block_{i}')(x)
        features = out.F
        xyz = out.C[:,1:] * self.voxel_size

        sampled_features_batch = []
        sampled_xyz_batch = []
        sample_inds_batch = []
        for batch_id in range(batch_num):
            batch_id_list = out.C[:,0]
            batch_indices = torch.where(batch_id_list==batch_id)

            # select features and xyz
            features_batch = features[batch_indices]
            xyz_batch = xyz[batch_indices]

            # permute features  (Batch),Dims,Number
            features_batch = features_batch.transpose(1,0).contiguous()

            # sample to the same number
            ## Downsampling voxel points into specific number; (N=2048)
            xyz_batch_squ = xyz_batch.unsqueeze(0)
            features_batch_squ = features_batch.unsqueeze(0)
            
            if self.random_fps:
                new_idx = torch.randperm(xyz_batch_squ.shape[1])
                # bs,ndim,nvoxels = features_batch_squ.shape
                features_batch_squ = features_batch_squ[:,:,new_idx]
                xyz_batch_squ = xyz_batch_squ[:,new_idx,:]

            sampled_xyz, sampled_features, sample_inds \
                = self.fps_module(xyz_batch_squ, features_batch_squ, self.npoint)
            # store the sampled ones
            sampled_features_batch.append(sampled_features)
            sampled_xyz_batch.append(sampled_xyz)
            sample_inds_batch.append(sample_inds)

        enc_features = torch.cat(sampled_features_batch)
        enc_xyz = torch.cat(sampled_xyz_batch)       
        enc_inds = torch.cat(sample_inds_batch)
        # xyz: batch x allpoints x 3
        # features: None
        # enc_xyz: batch x npoints x 3
        # enc_features: batch x channel x npoints
        # enc_inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        enc_features = enc_features.permute(2, 0, 1)

        return enc_xyz, enc_features, enc_inds

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
                    
        bs,npoints,_ = enc_xyz.shape
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        enc_box_predictions = {}
        scene_size = point_cloud_dims[1]-point_cloud_dims[0]
        point_cls_logits = self.decoder.pointcls_heads(enc_features.permute(1,2,0).contiguous()).transpose(1, 2).reshape((bs,npoints,-1)).contiguous()
        class_idx = point_cls_logits.sigmoid().max(dim=-1)[1]
        if self.hard_anchor:
            # print("use hard anchor")
            size_per_class = enc_features.new_tensor(self.dataset_config.mean_size_arr_hard_anchor)
        else:
            size_per_class = enc_features.new_tensor(self.dataset_config.mean_size_arr)
        size_unnormalized = size_per_class[class_idx]
        query_xyz, query_embed, query_inds = self.get_query_embeddings(enc_xyz, enc_features, point_cloud_dims)

        enc_box_predictions["point_cls_logits"] = point_cls_logits
        enc_box_predictions["center_unnormalized"] = query_xyz #256 FPS; 1024 decproposal 
        enc_box_predictions["center_normalized"] = convert_unnorm2norm(query_xyz, point_cloud_dims)
        enc_box_predictions["size_unnormalized"] = size_unnormalized
        enc_box_predictions["size_normalized"] = convert_unnorm2norm(size_unnormalized, point_cloud_dims, with_offset=False)
        enc_box_predictions["box_corners"] =  self.decoder.box_processor.box_parametrization_to_corners(
                                                    enc_box_predictions["center_unnormalized"], enc_box_predictions["size_unnormalized"] , query_xyz.new_zeros((bs,query_xyz.shape[1])).float()
                                                    )   
        if not self.querypos_mlp:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = None

        enc_box_features = enc_features
        box_predictions = self.decoder(
            tgt, enc_features, query_xyz, enc_xyz, point_cloud_dims, 
            query_pos=query_embed, enc_box_predictions=enc_box_predictions,
            enc_box_features = enc_box_features,
        )[0]

        
        box_predictions['seed_inds'] = enc_inds
        box_predictions['seed_xyz'] = enc_xyz            
        if enc_box_predictions is not None:
            box_predictions["enc_outputs"] = enc_box_predictions
        return box_predictions

def convert_unnorm2norm(xyz_unnorm,point_cloud_dims,with_offset=True):
    scene_size = point_cloud_dims[1]-point_cloud_dims[0] #bs,3
    if with_offset:
        offset = point_cloud_dims[0].unsqueeze(1)
    else:
        offset = 0
    xyz_norm = (xyz_unnorm-offset)/scene_size.unsqueeze(1)
    return xyz_norm


def build_backbone(args):
    if args.use_color and args.xyz_color:
        if args.use_normals:
            point_dim = 9
        else:
            point_dim = 6
    else:
        if args.use_normals:
            point_dim = 6
        else:
            point_dim = 3
    preencoder = MinkResNet(
        depth=args.depth, 
        in_channels=point_dim,
        inplanes=args.inplanes,
        num_stages=args.num_stages,
        stem_bn=args.stem_bn,)
    return preencoder


def build_decoder(args, dataset_config):
    first_layer = FFNLayer(
        d_model=args.dec_dim,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )

    decoder_layer = GlobalDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
        pos_for_key=args.pos_for_key,
        args=args
    )
    
    decoder = TransformerDecoder(
        first_layer,
        decoder_layer, 
        dataset_config, 
        num_layers=args.dec_nlayers-1, 
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        mlp_norm=args.mlp_norm,
        mlp_act=args.mlp_act,
        mlp_sep=args.mlp_sep,
        pos_for_key=args.pos_for_key,
        num_queries=args.nqueries,
        cls_loss=args.cls_loss,
        is_bilable=args.is_bilable,
        q_content=args.q_content,
        return_intermediate=True,
        args=args      
    )
    return decoder


def build_vdetr(args, dataset_config):
    pre_encoder = build_backbone(args)
    encoder = None #we remove the transfoemer encoder of 3DETR
    decoder = build_decoder(args, dataset_config)
    model = ModelVDETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        num_queries=args.nqueries,
        querypos_mlp=args.querypos_mlp,
        minkowski=args.minkowski,
        inplane=args.inplanes,
        num_stages=args.num_stages,
        voxel_size=args.voxel_size,
        npoint=args.preenc_npoints,
        use_fpn=args.use_fpn,
        layer_idx=args.layer_idx,
        proj_nohid=args.proj_nohid,
        woexpand_conv=args.woexpand_conv,
        args=args,
    )
    return model
