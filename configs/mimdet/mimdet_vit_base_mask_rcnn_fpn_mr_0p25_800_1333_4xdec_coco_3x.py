from .mimdet_vit_base_mask_rcnn_fpn_mr_0p5_800_1333_4xdec_coco_3x import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
    mae_checkpoint,
)

model.backbone.bottom_up.sample_ratio = 0.25

train.output_dir = "output/mimdet_vit_base_mask_rcnn_fpn_mr_0p25_800_1333_4xdec_coco_3x"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
