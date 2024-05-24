# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train script"""
import os
import datetime as dt
from functools import reduce
from mindspore import save_checkpoint
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from src.tools.common import get_callbacks

from src.args import get_args
from src.model.factory import create_model
from src.tools.cell import cast_amp
#from src.tools.criterion import get_criterion_by_args, NetWithLoss
from src.tools.criterion2 import get_criterion_by_args_distill, NetWithLoss, NetWithLossEval
from src.tools.get_misc import set_device, load_pretrained, \
    get_train_one_step, get_directories, save_config
from src.tools.optimizer import get_optimizer_distill
from src.tools.evalcell import EvalCell

from src.data.imagenet import create_datasets
from src.model.layers.trans_layer import get_trans_layer
from mindspore.common.initializer import One, Normal
from mindspore import Tensor,float32
import numpy as np
def main():
    args = get_args()
    args.epochs = 4
    args.mode = ""
    set_seed(args.seed)

    if args.mode == 'GRAPH_MODE':
        mode = context.GRAPH_MODE
    else:
        mode = context.PYNATIVE_MODE
    context.set_context(
        mode=mode, device_target=args.device_target
    )
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_reduce_precision=True)
    rank = set_device(args.device_target, args.device_id)

    #args.model = 'midmap_deit_base_patch16_224'
    net = create_model(
        model_name=args.student_model,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path
    )

    trans_layer_att = get_trans_layer(args.teacher_embed_dim, net.blocks[0].embed_dim, loss_pos=args.loss_pos)
    trans_layer_map = get_trans_layer(args.teacher_embed_dim, net.blocks[0].embed_dim, loss_pos=args.loss_pos)

    args.student_net_embed_dim = net.embed_dim
    #cast_amp(net, args.amp_level, args)

    # 测试网络的可行性
    # x = Tensor(shape=(1,3,224,224), dtype=float32, init=One())
    # y, out_att, out_map, out_embed = net(x)

    print(
        'Number of parameters:',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * int(
            os.environ.get("DEVICE_NUM", 1)
        ) / 512.0
        args.lr = linear_scaled_lr

    criterion = get_criterion_by_args_distill(args)   #在这里引入teacher_model并定义蒸馏损失
    net_with_loss = NetWithLoss(net, trans_layer_att, trans_layer_map, criterion)

    train_dataset, val_dataset = create_datasets(args)
    batch_num = train_dataset.get_dataset_size()
    optimizer = get_optimizer_distill(args, net, trans_layer_att, trans_layer_map, batch_num)
    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    if args.finetune:
        load_pretrained(
            net, args.finetune, args.num_classes, args.exclude_epoch_state
        )
    eval_network = EvalCell(
        net, nn.CrossEntropyLoss(), args.amp_level in ["O2", "O3", "auto"]
    )
    # eval_network = NetWithLossEval(net, trans_layer_att, trans_layer_map, criterion)

    eval_indexes = [0, 1, 2]
    model = Model(
        net_with_loss,
        metrics={'acc', 'loss'},
        eval_network=eval_network,
        eval_indexes=eval_indexes,
    )

    summary_dir, ckpt_dir, best_ckpt_dir, last_ckpt_dir, prefix = get_directories(
        args.student_model,
        args.output_dir,
        rank=rank,
    )

    callbacks = get_callbacks(
        arch=args.student_model,
        rank=rank,
        train_data_size=train_dataset.get_dataset_size(),
        val_data_size=val_dataset.get_dataset_size(),
        ckpt_dir=ckpt_dir,
        best_ckpt_dir=best_ckpt_dir,
        summary_dir=summary_dir,
        ckpt_save_every_step=args.ckpt_save_every_step,
        ckpt_save_every_sec=args.ckpt_save_every_sec,
        ckpt_keep_num=args.ckpt_keep_num,
        print_loss_every=args.print_loss_every,
        collect_freq=1,
        collect_tensor_freq=10,
        collect_input_data=args.collect_input_data,
        keep_default_action=False,
    )
    save_config(args, os.path.join(args.output_dir,prefix))

    print("begin train")
    model.fit(
        int(args.epochs - args.start_epoch),
        train_dataset,
        val_dataset,
        callbacks=callbacks,
        dataset_sink_mode=not args.no_dataset_sink_mode,
        valid_dataset_sink_mode=not args.no_dataset_sink_mode and not args.model_ema,
    )
    net_path = os.path.join(last_ckpt_dir, 'Learngenepool' + str(args.blk_length) + ".ckpt")
    # save_config(args, os.path.join(args.output_dir, prefix))
    save_checkpoint(net, net_path)

    print("train success")


if __name__ == '__main__':
    main()
