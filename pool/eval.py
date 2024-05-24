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
"""Evaluation script."""

from functools import reduce

import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.args import get_args
from src.model.factory import create_model
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLossEval
from src.tools.get_misc import set_device, load_pretrained, \
    get_train_one_step
from src.tools.optimizer import get_optimizer
from src.model.factory import create_learngene_pool_eval,create_learngene_pool

from src.data.imagenet import create_datasets


def main():
    args = get_args()
    set_seed(args.seed)
    args.mode = ''
    args.blk_length = 6
    args.device_num = 1
    args.device_id = 0
    args.num_workers = 1
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
    set_device(args.device_target, args.device_id)

    # net = create_learngene_pool(args)
    net = create_learngene_pool_eval(args)

    cast_amp(net, args.amp_level, args)
    net.set_train(False)

    print(
        'Number of parameters:',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    criterion = nn.CrossEntropyLoss
    net_with_loss = NetWithLossEval(net, criterion)
    _, val_dataset = create_datasets(args)

    batch_num = val_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    eval_network = nn.WithEvalCell(
        net, nn.CrossEntropyLoss(), args.amp_level in ["O2", "O3", "auto"]
    )
    eval_indexes = [0, 1, 2]
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    loss_monitor_cb = ms.LossMonitor(args.print_loss_every)
    Loss = []
    Top1 = []
    Top5 = []
    print(f"=> begin eval")
    for i in range(len(net.stitch_cfg_ids)):
        results = model.eval(val_dataset, callbacks=[loss_monitor_cb])
        print(f"=> eval results: {results}")
        Loss.append(results['Loss'])
        Top1.append(results['Top1-Acc'])
        Top5.append(results['Top5-Acc'])
        net.choose = net.choose + 1
    print(f"=> eval success")
    print(Top1)
    s=0
    for i in Top1:
        s=s+i 
    print(s/len(Top1))
    s=0
    for i in Loss:
        s=s+i 
    print(s/len(Loss))



if __name__ == '__main__':
    main()
