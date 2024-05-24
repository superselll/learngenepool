from mindspore import nn
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

class EvalCell(nn.WithEvalCell):
    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super().__init__(network=network, loss_fn=loss_fn, add_cast_fp32=add_cast_fp32)

    def construct(self, data, label):
        outputs = self._network(data)
        if len(outputs) > 1:
            outputs = outputs[0]

        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label

class EvalCell2(nn.WithEvalCell):
    def __init__(self, network, loss_fn,trans_layer_att, trans_layer_map, add_cast_fp32=False):
        super().__init__(network=network, loss_fn=loss_fn, add_cast_fp32=add_cast_fp32)
        self.trans_layer_att = trans_layer_att
        self.trans_layer_map = trans_layer_map

    def construct(self, data, label):
        outputs = self._network(data)
        if len(outputs) > 1:
            outputs = outputs[0]

        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label
