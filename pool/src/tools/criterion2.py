import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import (
    functional,
    operations,
    LogSoftmax,
    KLDivLoss,
    Size
)
from src.model.factory import create_model
from src.model.layers.trans_layer import get_trans_layer
'''
class soft_cross_entropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self):
        super(soft_cross_entropy, self).__init__()
        self.mean_ops = operations.ReduceMean(keep_dims=False)
        self.sum_ops = operations.ReduceSum(keep_dims=False)
        self.log_softmax = operations.LogSoftmax()
        self.softmax = operations.Softmax()

    def construct(self, predicts, labels):
        student_likelihood = self.log_softmax(predicts, -1)
        targets_prob = self.softmax(labels, -1)
        return self.mean_ops((- targets_prob * student_likelihood))
'''


def soft_cross_entropy(predicts, targets):

    student_likelihood = operations.LogSoftmax()(predicts)
    targets_prob = operations.Softmax()(targets)
    return operations.ReduceMean(keep_dims=False)((- targets_prob * student_likelihood))



def get_criterion_by_args_distill(args):
    criterion = get_criterion_distill(
        teacher_path=args.teacher_path,
        teacher_model=args.teacher_model,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        temperature = args.temperature,
        loss_pos=args.loss_pos,
        alpha = args.alpha,
        distill_loss = args.distill_loss,
    )
    return criterion

def get_criterion_distill(
        teacher_path,
        teacher_model,
        num_classes,
        drop_rate,
        drop_path_rate,
        temperature,
        loss_pos,
        alpha,
        distill_loss,
):
    MSE_loss = nn.MSELoss()
    CE_loss = nn.CrossEntropyLoss()
    teacher_net = create_model(
        model_name=teacher_model,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        checkpoint_path=teacher_path,
    )
    teacher_net.set_train(False)

    criterion = DistillationLoss(
        teacher_net,
        MSE_loss,
        CE_loss,
        soft_cross_entropy,
        temperature,
        loss_pos,
        alpha,
        distill_loss
    )

    return criterion

class DistillationLoss(LossBase):
    def __init__(self, teacher_net,
                 MSE_loss, CE_loss, soft_cross_entropy, temperature, loss_pos, alpha, distill_loss):
        super().__init__()
        self.teacher_net = teacher_net
        self.MSE_loss = MSE_loss
        self.CE_loss = CE_loss
        self.soft_cross_entropy = soft_cross_entropy,
        self.temperature = temperature,
        self.loss_pos = loss_pos
        self.alpha = alpha
        self.distill_loss = distill_loss

    def construct(self, data, predict, label, trans_layer_att, trans_layer_map, stu_model):
        stu_out, stu_out_atts, stu_out_maps, stu_out_embed = predict
        tea_out, tea_out_atts, tea_out_maps, tea_out_embed = self.teacher_net(data)
        num_stu_blk = len(stu_out_atts)  # 9，因为student_model.depth=9,所以有三个block,产生三个att之后的输出

        cls_loss = self.CE_loss(stu_out, label)  # 和目标之间的分类损失
        logits_loss = soft_cross_entropy(stu_out / self.temperature, tea_out / self.temperature)  # 全连接层输出的特征损失
        att_loss = 0.
        rep_loss = 0.
        if self.loss_pos == '[end]':
            tea_out_att = trans_layer_att[0](tea_out_atts[-1])
            tea_out_map = trans_layer_map[0](tea_out_maps[-1])
            att_loss = self.MSE_loss(stu_out_atts[-1], tea_out_att)  # 最后一个blk中Attention层输出特征的损失
            rep_loss = self.MSE_loss(stu_out_maps[-1], tea_out_map)  # 最后一个blk输出特征的损失
        elif self.loss_pos == '[mid,end]':
            stu_point = [int((num_stu_blk / 3) * 2 - 1), int((num_stu_blk / 3) * 3 - 1)]
            tea_point = [7, 11]
            for id, i, j in zip([0, 1], stu_point, tea_point):
                tea_out_att = trans_layer_att[id](tea_out_atts[j])
                tea_out_map = trans_layer_map[id](tea_out_maps[j])
                att_loss += self.MSE_loss(stu_out_atts[i], tea_out_att)
                rep_loss += self.MSE_loss(stu_out_maps[i], tea_out_map)
        elif self.loss_pos == '[front,end]':
            stu_point = [int((num_stu_blk / 3) - 1), int((num_stu_blk / 3) * 2 - 1),
                         int((num_stu_blk / 3) * 3 - 1)]  # [2,5,8]
            tea_point = [3, 7, 11]
            for id, i, j in zip([0, 1, 2], stu_point, tea_point):
                tea_out_att = trans_layer_att[id](tea_out_atts[j])
                tea_out_map = trans_layer_map[id](tea_out_maps[j])
                att_loss += self.MSE_loss(stu_out_atts[i], tea_out_att)
                rep_loss += self.MSE_loss(stu_out_maps[i], tea_out_map)
        loss = 0.
        if self.distill_loss == 'logits':
            alpha = 0.5
            loss = alpha * cls_loss + (1 - alpha) * (logits_loss)
        elif self.distill_loss == 'logits+rep':
            alpha = 0.5
            loss = alpha * cls_loss + (1 - alpha) * (logits_loss + rep_loss)
        elif self.distill_loss == 'all':
            loss = self.alpha * cls_loss + (1 - self.alpha) * (logits_loss + rep_loss + att_loss)
        '''
        networks = [stu_model, trans_layer_att, trans_layer_map]
        params = []
        for network in networks:
            for x in network.trainable_params():
                params.append(x)
        out = ops.clip_by_global_norm(params, 1.0)
        '''
        return loss

class NetWithLoss(nn.Cell):
    """
    NetWithLoss: Only support Network with Classification.
    """

    def __init__(self, model, trans_layer_att, trans_layer_map, criterion):
        super(NetWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion
        self.trans_layer_att = trans_layer_att
        self.trans_layer_map = trans_layer_map

    def construct(self, *inputs, **kwargs):
        data = inputs[0]
        label = inputs[1]
        predict = self.model(data)
        loss = self.criterion(data, predict, label, self.trans_layer_att, self.trans_layer_map, self.model)
        return loss

class NetWithLossEval(nn.Cell):
    """
    NetWithLoss: Only support Network with Classification.
    """

    def __init__(self, model, trans_layer_att, trans_layer_map, criterion):
        super(NetWithLossEval, self).__init__()
        self.model = model
        self.criterion = criterion
        self.trans_layer_att = trans_layer_att
        self.trans_layer_map = trans_layer_map

    def construct(self, *inputs, **kwargs):
        data = inputs[0]
        label = inputs[1]
        predict = self.model(data)
        loss = self.criterion(data, predict, label, self.trans_layer_att, self.trans_layer_map, self.model)
        return loss, predict[0], label