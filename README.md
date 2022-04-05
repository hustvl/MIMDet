<div align="center">
<h1>MIMDet &#127917;</h1>
<span><font size="4", >Taming and Unleashing Vanilla Vision Transformer
with Masked Image Modeling for Object Detection</font></span>
</div>


## Introduction


## Models and Main Results

### Mask R-CNN
| Model | Sample Ratio | Schedule | Aug | box mAP | mask mAP | #params | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MIMDet-ViT-B | 0.25 | 3x | [480-800, 1333] w/crop | 49.9 | 44.7 | 127.56M | [config](configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_mr_0p25_800_1333_4xdec_coco_3x.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/mimdet_vit_base_mask_rcnn_fpn_sr_0p25_800_1333_4xdec_coco_3x.pth) |
| MIMDet-ViT-B | 0.5 | 3x | [480-800, 1333] w/crop | 51.5 | 46.0 | 127.56M | [config](configs/mimdet/mimdet_vit_base_mask_rcnn_fpn_mr_0p5_800_1333_4xdec_coco_3x.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/mimdet_vit_base_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x.pth) |
| MIMDet-ViT-L | 0.5 | 3x | [480-800, 1333] w/crop | 53.3 | 47.5 | 345.27M | [config](configs/mimdet/mimdet_vit_large_mask_rcnn_fpn_mr_0p5_800_1333_4xdec_coco_3x.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/mimdet_vit_large_mask_rcnn_fpn_sr_0p5_800_1333_4xdec_coco_3x.pth) |
| Benchmarking-ViT-B | - | 25ep | [1024, 1024] LSJ(0.1-2) | 48.0 | 43.0 | 118.67M | [config](configs/benchmarking/benchmarking_mask_rcnn_base_FPN_25ep_LSJ_mae.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/benchmarking_mask_rcnn_base_FPN_25ep_LSJ_mae.pth) |
| Benchmarking-ViT-B | - | 50ep | [1024, 1024] LSJ(0.1-2) | 50.2 | 44.9 | 118.67M | [config](configs/benchmarking/benchmarking_mask_rcnn_base_FPN_50ep_LSJ_mae.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/benchmarking_mask_rcnn_base_FPN_50ep_LSJ_mae.pth) |
| Benchmarking-ViT-B | - | 100ep | [1024, 1024] LSJ(0.1-2) | 50.4 | 44.9 | 118.67M | [config](configs/benchmarking/benchmarking_mask_rcnn_base_FPN_100ep_LSJ_mae.py) | [github](https://github.com/hustvl/storage/releases/download/v1.0.0/benchmarking_mask_rcnn_base_FPN_100ep_LSJ_mae.pth) |

## Installation

- Clone
```
git clone https://github.com/hustvl/MIMDet.git
cd MIMDet
```

- Create a conda virtual environment and activate it:
```
conda create -n mimdet python=3.9
conda activate mimdet
```

* This repo is baed on [`Detectron2==0.6`](https://github.com/facebookresearch/detectron2), installation follow [link](https://detectron2.readthedocs.io/tutorials/install.html).
* This repo is baed on [`timm==0.4.12`](https://github.com/rwightman/pytorch-image-models), installation follow [link](https://fastai.github.io/timmdocs/).

## Inference

```
# inference
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --eval-only train.init_checkpoint=<MODEL_PATH>

# inference with 100% sample ratio
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --eval-only train.init_checkpoint=<MODEL_PATH> model.backbone.bottom_up.sample_ratio=1.0
```

## Training

```
# single-machine training
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> mae_checkpoint.path=<MAE_MODEL_PATH>

# multi-machine training
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --num-machines <MACHINE_NUM> --master_addr <MASTER_ADDR> --master_port <MASTER_PORT> mae_checkpoint.path=<MAE_MODEL_PATH>
```

## License

MIMDet is released under the [MIT License](LICENSE).
