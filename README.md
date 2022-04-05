<div align="center">
<h1>MIMDet &#127917;</h1>
<span><font size="4", >Taming and Unleashing Vanilla Vision Transformer
with Masked Image Modeling for Object Detection</font></span>
</div>


## Introduction


## Models and Main Results


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
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM> --eval-only train.init_checkpoint=<CHECKPOINT_PATH>
```

## Training

```
# single-machine training
python lazyconfig_train_net.py --config-file <CONFIG_FILE> --num-gpus <GPU_NUM>

# multi-machine training
```

## License

MIMDet is released under the [MIT License](LICENSE).
