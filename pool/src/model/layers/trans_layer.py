import mindspore.nn as nn


class TransformerLayer(nn.Cell):
    def __init__(self, tea_dim=None, stu_dim=None):
        super().__init__()

        self.tea_dim = tea_dim
        self.stu_dim = stu_dim
        self.transform = nn.Dense(self.tea_dim, self.stu_dim)

    def construct(self, x):
        x = self.transform(x)
        return x

def get_trans_layer(tea_dim=None, stu_dim=None, loss_pos='[end]'):
    trans_layer_list = nn.CellList()
    if loss_pos == '[end]':
        length = 1
    elif loss_pos == '[mid,end]':
        length = 2
    elif loss_pos == '[front,end]':
        length = 3
    for len in range(length):
        trans_layer_list.append(TransformerLayer(tea_dim, stu_dim))
    return trans_layer_list