<div align="center">
<h1>MIMDet &#127917;</h1>
<h3>Unleashing Vanilla Vision Transformer
with Masked Image Modeling for Object Detection</h3>


[Yuxin Fang](https://bit.ly/YuxinFang_GoogleScholar)<sup>1</sup> \*, [Shusheng Yang](https://scholar.google.com/citations?user=v6dmW5cntoMC&hl=en)<sup>1</sup> \*, [Shijie Wang](https://github.com/simonJJJ)<sup>1</sup> \*, [Yixiao Ge](https://geyixiao.com/)<sup>2</sup>, [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)<sup>2</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>1 :email:</sup>,
 
<sup>1</sup> [School of EIC, HUST](http://eic.hust.edu.cn/English/Home.htm), <sup>2</sup> [ARC Lab, Tencent PCG](https://arc.tencent.com/en/index).

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.


</div>



## Introduction


<p align="center">
<img src="MIMDet.png" width=80%>
</p>

This repo provides code and pretrained models for **MIMDet** (**M**asked **I**mage **M**odeling for **Det**ection).
* MIMDet is a simple framekwork that enables a MIM pretrained vanilla ViT to perform high-performance object-level understanding, e.g, object detection and instance segmentation.
* In MIMDet, a MIM pre-trained vanilla ViT encoder can work surprisingly well in the challenging object-level recognition scenario even with randomly sampled *partial* observations, e.g., only 25%~50% of the input embeddings.
* In order to construct multi-scale representations for object detection, a *randomly initialized* compact convolutional stem supplants the pre-trained large kernel patchify stem, and its intermediate features can naturally serve as the higher resolution inputs of a feature pyramid without upsampling. While the pre-trained ViT is only regarded as the third-stage of our detector's backbone instead of the whole feature extractor, resulting in a ConvNet-ViT *hybrid* architecture.
* MIMDet-Base obtains **51.5 box AP** and **46.0 mask AP** on COCO.

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

**Notes**:

- Benchmarking-ViT-B is an unofficial implementation of [Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/abs/2111.11429)
- The configuration & results of MIMDet-ViT-L are still under-tuned.

## Installation

### Prerequisites
* Linux
* Python 3.7+
* CUDA 10.2+
* GCC 5+

### Prepare

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

* Install [`Detectron2==0.6`](https://github.com/facebookresearch/detectron2), follow [d2 doc](https://detectron2.readthedocs.io/tutorials/install.html).
* Install [`timm==0.4.12`](https://github.com/rwightman/pytorch-image-models), follow [timm doc](https://fastai.github.io/timmdocs/).
* Install [`einops`](https://github.com/arogozhnikov/einops), follow [einops repo](https://github.com/arogozhnikov/einops#installation--).
* Prepare MS-COCO dataset, follow [d2_doc](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html).

## Inference

```
# inference
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --eval-only train.init_checkpoint=<MODEL_PATH>

# inference with 100% sample ratio
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --eval-only train.init_checkpoint=<MODEL_PATH> model.backbone.bottom_up.sample_ratio=1.0
```

## Training

Downloading MAE full pretrained [ViT-B Model](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth) and [ViT-L Model](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth), follow [MAE repo-issues-8](https://github.com/facebookresearch/mae/issues/8).
```
# single-machine training
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> mae_checkpoint.path=<MAE_MODEL_PATH>

# multi-machine training
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --num-machines <MACHINE_NUM> --master_addr <MASTER_ADDR> --master_port <MASTER_PORT> mae_checkpoint.path=<MAE_MODEL_PATH>
```

## Acknowledgement
This project is based on [MAE](https://github.com/facebookresearch/mae), [Detectron2](https://github.com/facebookresearch/detectron2) and [timm](https://github.com/rwightman/pytorch-image-models). Thanks for their wonderful works.

## License

MIMDet is released under the [MIT License](LICENSE).
