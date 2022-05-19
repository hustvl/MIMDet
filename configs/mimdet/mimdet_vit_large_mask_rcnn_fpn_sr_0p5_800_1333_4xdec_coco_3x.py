from .mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
    mae_checkpoint,
)

model.backbone.bottom_up.encoder.embed_dim = 1024
model.backbone.bottom_up.encoder.depth = 24
model.backbone.bottom_up.encoder.num_heads = 16
model.backbone.bottom_up.encoder.dpr = 0.4
model.backbone.bottom_up.decoder.embed_dim = 1024
model.backbone.bottom_up._out_feature_channels = [256, 512, 512, 512]

optimizer.params.multiplier = 3.5

train.output_dir = "output/mimdet_vit_large_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
