# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# Swin Transformer: https://github.com/microsoft/Swin-Transformer
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block

from utils.pos_embed import (
    get_2d_sincos_pos_embed,
    interpolate_pos_embed,
    interpolate_pos_embed_online,
)

__all__ = ["ConvStem", "MIMDetEncoder", "MIMDetDecoder", "MIMDetBackbone"]


class ConvStem(nn.Module):
    """ConvStem, from Early Convolutions Help Transformers See Better, Tete et
    al.
    https://arxiv.org/abs/2106.14881
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=4,
        norm_layer=None,
        checkpointing=False,
    ):
        super().__init__()

        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.depth = depth
        self.checkpointing = checkpointing

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // (2 ** (depth - 1))
        for idx in range(depth):
            stage_list = [
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, output_dim, eps=1e-6),
                nn.GELU(),
            ]
            if idx == depth - 1:
                stage_list.append(nn.Conv2d(output_dim, embed_dim, kernel_size=1))
            stage = nn.Sequential(*stage_list)
            input_dim = output_dim
            output_dim *= 2
            stem.append(stage)
        self.proj = nn.ModuleList(stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        outputs = []
        for i, stage in enumerate(self.proj):
            if self.checkpointing and x.requires_grad:
                x = cp.checkpoint(stage, x)
            else:
                x = stage(x)
            if i >= 1:
                if i == (len(self.proj) - 1):
                    outputs.append(self.norm(x))
                else:
                    outputs.append(x)
        return outputs


class MIMDetEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dpr=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained=None,
        checkpointing: bool = False,
    ):
        super().__init__()

        self.checkpointing = checkpointing
        self.patch_embed = ConvStem(
            img_size, patch_size, in_chans, embed_dim, checkpointing=checkpointing
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )

        dpr = [
            x.item() for x in torch.linspace(0, dpr, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if pretrained:
            checkpoint_model = torch.load(pretrained, map_location="cpu")["model"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                if "encoder" in k:
                    new_checkpoint_model[k.replace("encoder.", "")] = v
                elif "module" in k:
                    new_checkpoint_model[k.replace("module.", "")] = v
                else:
                    new_checkpoint_model[k] = v
            interpolate_pos_embed(self, new_checkpoint_model, "pos_embed")
            print(self.load_state_dict(new_checkpoint_model, strict=False))
            print(f"Loading ViT Encoder pretrained weights from {pretrained}.")
        else:
            print("Loading ViT Encoder pretrained weights from scratch.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, sample_ratio, masks):
        N, L, D = x.shape
        masks_flatten = masks[0].flatten(1)
        assert masks_flatten.shape[1] == L
        len_keep = int(L * sample_ratio)

        noise = torch.rand(N, L, device=x.device)
        noise = noise.masked_fill(masks_flatten, 100)

        # sort noise for each sample
        ids_keep = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_keep, dim=1)

        # keep the first subset
        ids_keep = ids_keep[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones(
                (N, H, W), dtype=torch.bool, device=device
            )
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.patch_embed.patch_size[0])),
                    : int(np.ceil(float(w) / self.patch_embed.patch_size[1])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

    def forward(self, imgs, sample_ratio):
        outputs = self.patch_embed(imgs.tensor)
        x = outputs[-1]
        H, W = x.shape[-2:]
        masks = self.mask_out_padding([x.shape], imgs.image_sizes, imgs.tensor.device)
        x = x.flatten(2).transpose(1, 2)
        pos_embed = interpolate_pos_embed_online(
            self.pos_embed, self.patch_embed.grid_size, (H, W), 1
        )[:, 1:, :]
        x = x + pos_embed

        x, ids_restore = self.random_masking(x, sample_ratio, masks)

        if self.checkpointing and x.requires_grad:
            for blk in self.blocks:
                x = cp.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)
        outputs.append(x)
        x = outputs

        return x, ids_restore, (H, W)


class MIMDetDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        decoder_embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        dpr=0.0,
        norm_layer=nn.LayerNorm,
        pretrained=None,
        checkpointing=False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.conv_feat_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.initialize_weights(pretrained)

    def initialize_weights(self, pretrained):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        if pretrained:
            checkpoint_model = torch.load(pretrained, map_location="cpu")["model"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                if "encoder" in k:
                    new_checkpoint_model[k.replace("encoder.", "")] = v
                elif "module" in k:
                    new_checkpoint_model[k.replace("module.", "")] = v
                else:
                    new_checkpoint_model[k] = v
            interpolate_pos_embed(self, new_checkpoint_model, "decoder_pos_embed")
            print(self.load_state_dict(new_checkpoint_model, strict=False))
            print(f"Loading ViT Decoder pretrained weights from {pretrained}.")
        else:
            print("Loading ViT Decoder pretrained weights from scratch.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore, new_size, conv_feats):
        x = self.decoder_embed(x)
        B, L, C = x.shape
        conv_feats = conv_feats.flatten(2).permute(0, 2, 1)
        conv_feats = self.conv_feat_proj(conv_feats)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        conv_feats = torch.gather(conv_feats, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))
        x_ = torch.cat([x, conv_feats[:, L:, :]], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )  # unshuffle

        pos_embed = interpolate_pos_embed_online(
            self.decoder_pos_embed, self.grid_size, new_size, 1
        )[:, 1:, :]

        x = x + pos_embed
        if self.checkpointing and x.requires_grad:
            for blk in self.decoder_blocks:
                x = cp.checkpoint(blk, x)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        if self.checkpointing and x.requires_grad:
            x = cp.checkpoint(self.decoder_norm, x)
        else:
            x = self.decoder_norm(x)
        return x.transpose(1, 2).reshape(B, C, *new_size)


class MIMDetBackbone(Backbone):
    def __init__(
        self,
        encoder,
        decoder,
        sample_ratio,
        size_divisibility,
        _out_feature_channels=[192, 384, 512, 512],
        _out_feature_strides=[4, 8, 16, 32],
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sample_ratio = sample_ratio
        self._size_divisibility = size_divisibility
        self._out_feature_strides = _out_feature_strides
        self._out_feature_channels = _out_feature_channels
        self._out_features = ["c2", "c3", "c4", "c5"]

    def forward(self, x):
        latent, ids_restore, new_size = self.encoder(x, self.sample_ratio)
        latent[-1] = self.decoder(latent[-1], ids_restore, new_size, latent[-2])
        return {
            "c2": latent[0],
            "c3": latent[1],
            "c4": latent[-1],
            "c5": F.max_pool2d(latent[-1], 2),
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[i],
                stride=self._out_feature_strides[i],
            )
            for i, name in enumerate(self._out_features)
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility
