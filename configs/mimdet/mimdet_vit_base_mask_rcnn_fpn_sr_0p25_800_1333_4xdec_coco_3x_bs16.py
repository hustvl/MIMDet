from .mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
    mae_checkpoint,
)

model.backbone.bottom_up.sample_ratio = 0.25

dataloader.train.total_batch_size = 16
optimizer.params.base_lr *= 0.5 # scale lr
optimizer.lr *= 0.5 # scale lr
train.checkpointer.period = int(120000 / 16)
train.eval_period = int(120000 / 16)
train.max_iter = int(120000 / 16 * 36)
lr_multiplier.scheduler.milestones = [int(120000 / 16 * 27), int(120000 / 16 * 33)]
lr_multiplier.scheduler.num_updates = int(120000 / 16 * 36)

train.output_dir = "output/mimdet_vit_base_mask_rcnn_fpn_sr_0p25_800_1333_4xdec_coco_3x_bs16"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
