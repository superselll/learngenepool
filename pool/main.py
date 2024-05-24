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
from functools import reduce

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore import save_checkpoint
from src.tools.common import get_callbacks

from src.args import get_args
from src.model.factory import create_learngene_pool
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion_by_args, NetWithLoss, NetWithLossEval
from src.tools.get_misc import set_device, load_pretrained, \
    get_train_one_step, get_directories, save_config
from src.tools.optimizer import get_optimizer
from src.tools.evalcell import EvalCell
from src.model.factory import create_learngene_pool_eval,create_learngene_pool
from src.data.imagenet import create_datasets
from mindspore.common.initializer import One, Normal
from mindspore import Tensor,float32

import mindspore
import numpy as np
from mindspore import Tensor, ops
def main():

    args = get_args()
    args.mode = ""
    args.device_target = "GPU"
    args.batch_size = 32
    args.device_num = 1
    args.device_id = 0
    args.blk_length = 6
    args.num_workers = 1
    args.num_classes = 1000
    args.start_epoch = 0
    args.epochs = 100
    args.ckpt_save_every_step = 125000
    args.ckpt_keep_num = 10
    args.sched = None
    args.warmup_epochs=0
    args.lr = 5e-5
    args.distillation_type = 'none'
    set_seed(args.seed)
    if args.mode == "GRAPH_MODE":
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

    net = create_learngene_pool(args)
    # 若使用main训练好的learngenepool的权重 使用create_learngene_pool_eval
    # net = create_learngene_pool_eval(args)

    #cast_amp(net, args.amp_level, args)
    print(
        'Number of parameters:',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    '''
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * int(
            os.environ.get("DEVICE_NUM", 1)
        ) / 512.0
        args.lr = linear_scaled_lr
    '''
    criterion = get_criterion_by_args(args)   #在这里引入teacher_model并定义蒸馏损失
    net_with_loss = NetWithLoss(net, criterion)


    train_dataset, val_dataset = create_datasets(args)

    batch_num = train_dataset.get_dataset_size()

    optimizer = get_optimizer(args, net, batch_num)
    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    if args.finetune:
        load_pretrained(
            net, args.finetune, args.num_classes, args.exclude_epoch_state
        )
    eval_network = nn.WithEvalCell(
        net, nn.CrossEntropyLoss(), args.amp_level in ["O2", "O3", "auto"]
    )
    eval_indexes = [0, 1, 2]
    model = Model(
        net_with_loss,
        metrics={'acc', 'loss'},
        eval_network=eval_network,
        eval_indexes=eval_indexes,
    )
    summary_dir, ckpt_dir, best_ckpt_dir, last_ckpt_dir, prefix = get_directories(
        'Learngenepool' + str(args.blk_length),
        args.output_dir,
        rank=rank,
    )
    callbacks = get_callbacks(
        arch='Learngenepool' + str(args.blk_length),
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
    print("begin train")
    model.fit(
        int(args.epochs - args.start_epoch),
        train_dataset,
        val_dataset,
        callbacks=callbacks,
        dataset_sink_mode=not args.no_dataset_sink_mode,
        valid_dataset_sink_mode=False  # not args.no_dataset_sink_mode and not args.model_ema,
    )
    net_path = os.path.join(last_ckpt_dir, 'Learngenepool' + str(args.blk_length) + ".ckpt")
    # save_config(args, os.path.join(args.output_dir, prefix))
    save_checkpoint(net, net_path)
    print("train success")


if __name__ == '__main__':
    main()
