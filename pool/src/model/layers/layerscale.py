import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import Ones

class LayerScale(nn.Cell):

    def __init__(self, dim, init_values=1e-5, inplace: bool = False):
        super(LayerScale, self).__init__()
        self.inplace = inplace
        self.gamma = ms.Parameter(init_values * Ones()(dim), ms.float32)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return ms.ops.inplace_update(x, x*self.gamma, range(0, x.shape[0])) if self.inplace else x * self.gamma