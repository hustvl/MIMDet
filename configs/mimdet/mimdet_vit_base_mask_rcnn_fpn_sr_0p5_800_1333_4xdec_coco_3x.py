from functools import partial
import torch
import torch.nn as nn

import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    StandardROIHeads,
)

from models import MIMDetBackbone, MIMDetDecoder, MIMDetEncoder

from ..coco import dataloader
from ..common import (
    GeneralizedRCNNImageListForward,
    get_fpn_model_parameters,
    lr_multiplier,
    train,
    mae_checkpoint,
)

model = L(GeneralizedRCNNImageListForward)(
    lsj_postprocess=False,
    backbone=L(FPN)(
        bottom_up=L(MIMDetBackbone)(
            encoder=L(MIMDetEncoder)(
                img_size=800,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dpr=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                pretrained="${mae_checkpoint.path}",
                checkpointing=True,
            ),
            decoder=L(MIMDetDecoder)(
                img_size="${..encoder.img_size}",
                patch_size="${..encoder.patch_size}",
                embed_dim=768,
                decoder_embed_dim=512,
                depth=4,
                num_heads=16,
                mlp_ratio=4.0,
                pretrained="${..encoder.pretrained}",
                checkpointing=True,
            ),
            sample_ratio=0.5,
            size_divisibility=32,
            _out_feature_channels=[192, 384, 512, 512],
        ),
        in_features=["c2", "c3", "c4", "c5"],
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    input_format="RGB",
)

optimizer = L(torch.optim.AdamW)(
    params=L(get_fpn_model_parameters)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay=0.1,
        weight_decay_norm=0.0,
        base_lr="${..lr}",
        skip_list=("pos_embed", "decoder_pos_embed"),
        multiplier=2.0,
    ),
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

# dataloader
dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomApply)(
        tfm_or_aug=L(T.AugmentationList)(
            augs=[
                L(T.ResizeShortestEdge)(
                    short_edge_length=(400, 500, 600), sample_style="choice"
                ),
                L(T.RandomCrop)(crop_type="absolute_range", crop_size=(384, 600)),
            ]
        ),
        prob=0.5,
    ),
    L(T.ResizeShortestEdge)(
        short_edge_length=(480, 512, 544, 576, 608, 640, 736, 768, 800),
        sample_style="choice",
        max_size=1333,
    ),
]
dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
]

# batch size, lr & schedules
dataloader.train.total_batch_size = 64
train.checkpointer.period = int(120000 / 64)
train.eval_period = int(120000 / 64)
train.max_iter = int(120000 / 64 * 36)
lr_multiplier.scheduler.milestones = [int(120000 / 64 * 27), int(120000 / 64 * 33)]
lr_multiplier.scheduler.num_updates = int(120000 / 64 * 36)
lr_multiplier.warmup_length = 0.25 / 36  # warmup 1/4 epochs
lr_multiplier.warmup_factor = 0.0  # warmup from 0. * base_lr

# update stronger mask rcnn
# following the convolutions in FPN with sync batch normalization (syncbn)
model.backbone.norm = "SyncBN"
# using two convolutional layers in the region proposal network (RPN) instead of one
model.proposal_generator.head.conv_dims = [-1, -1]
# using four convolutional layers with GN followed by one linear layer for the region-of-interest (RoI) classiﬁcation and box regression head instead of a two-layer MLP without normalization
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]
# following the convolutions in the standard mask head with syncbn
model.roi_heads.box_head.conv_norm = (
    model.roi_heads.mask_head.conv_norm
) = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")

train.output_dir = "output/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x"
