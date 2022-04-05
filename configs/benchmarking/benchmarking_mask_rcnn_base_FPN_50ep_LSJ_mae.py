from .benchmarking_mask_rcnn_base_FPN_100ep_LSJ_mae import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = train.max_iter // 2  # 100ep -> 50ep

lr_multiplier.warmup_length *= 2

train.output_dir = "output/benchmarking_mask_rcnn_base_FPN_50ep_LSJ_mae"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
