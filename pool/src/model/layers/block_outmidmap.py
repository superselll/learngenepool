from functools import partial
import mindspore.nn as nn
from .block import Block

class Block_OutMidMap(Block):
    """
    The Block layer
    OutPut Four Params
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=partial(nn.GELU, approximate=False),
                 norm_layer=nn.LayerNorm):
        super().__init__(dim=dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop,
                         attn_drop=attn_drop,drop_path=drop_path,act_layer=act_layer,norm_layer=norm_layer)
        self.embed_dim = dim

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        out_att = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(out_att))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, out_att