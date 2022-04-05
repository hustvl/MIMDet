from .benchmarking_mask_rcnn_base_FPN_100ep_LSJ_mae import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = train.max_iter // 4  # 100ep -> 25ep

lr_multiplier.warmup_length *= 4

train.output_dir = "output/benchmarking_mask_rcnn_base_FPN_25ep_LSJ_mae"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
