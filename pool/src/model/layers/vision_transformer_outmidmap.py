from functools import partial

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Zeros

from mindspore.ops.function import concat
from mindspore.ops import Tile

from .patch_embed import PatchEmbed
from .block_outmidmap import Block_OutMidMap
from .weights_init import trunc_normal_
from .custom_identity import CustomIdentity

from .vision_transformer import VisionTransformer

class ViT_OutMidMap(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 act_mlp_layer=partial(nn.GELU, approximate=False)
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            raise NotImplementedError(
                'This Layer was not iimplementes because all models from deit does not use it'
            )
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(
            Zeros()((1, 1, embed_dim), ms.float32)
        )
        self.pos_embed = ms.Parameter(
            Zeros()((1, num_patches + 1, embed_dim), ms.float32)
        )
        self.tile = Tile()
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block = Block_OutMidMap
        self.blocks = nn.CellList([
            self.block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_mlp_layer
            )
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,))

        # Classifier head
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else CustomIdentity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self.cells(), self._init_weights)

    def apply(self, layer, fn):
        for l_ in layer:
            if hasattr(l_, 'cells') and len(l_.cells()) >= 1:
                self.apply(l_.cells(), fn)
            else:
                fn(l_)
    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                constant_init = ms.common.initializer.Constant(value=0)
                constant_init(m.bias)


    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else CustomIdentity()

    def forward_features(self, x):
        """Forward features"""
        b = x.shape[0]
        #patch_emd and pos_emd
        x = self.patch_embed(x)
        cls_tokens = self.tile(self.cls_token, (b, 1, 1))
        x = concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        #forward_block
        out_att = []
        out_map = []
        out_embed = x  # get embedding features
        for blk in self.blocks:
            x, att = blk(x)
            out_att.append(att)  # get attention maps
            out_map.append(x)  # get hidden features
        x = self.norm(x)
        return x[:, 0], out_att, out_map, out_embed

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x, out_atts, out_maps, out_embed = self.forward_features(x)
        out = self.head(x)
        return out, out_atts, out_maps, out_embed