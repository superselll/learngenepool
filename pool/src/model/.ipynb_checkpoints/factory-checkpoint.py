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

import mindspore as ms


from src.model.layers.deit import (
    deit_base_patch16_224,
    deit_base_distilled_patch16_224,
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_tiny_distilled_patch16_224,
    deit_small_distilled_patch16_224,
    deit_base_patch16_384,
    deit_base_distilled_patch16_384,
    midmap_deit_tiny_patch16_224,
    midmap_deit_tiny3_patch16_224,
    midmap_deit_tiny6_patch16_224,
    midmap_deit_tiny9_patch16_224,
    midmap_deit_small_patch16_224,
    midmap_deit_small3_patch16_224,
    midmap_deit_small6_patch16_224,
    midmap_deit_small9_patch16_224,
    midmap_deit_base_patch16_224,
    midmap_deit_base3_patch16_224,
    midmap_deit_base6_patch16_224,
    midmap_deit_base9_patch16_224,
    deit_base6_patch16_224,
    deit_small6_patch16_224,
    deit_tiny6_patch16_224,
    deit_base9_patch16_224,
    deit_small9_patch16_224,
    deit_tiny9_patch16_224,
)

from src.model.layers.regnet import regnety_160
from src.model.layers.learngene_instances import init_LearngenePool
from src.model.layers.learngenepool import LearngenePool
from src.model.serialization import load_param_into_net

def get_model_by_name(model_name, **kwargs):
    """get network by name and initialize it"""

    models = {
        'deit_base_patch16_224': deit_base_patch16_224,
        'deit_base_distilled_patch16_224': deit_base_distilled_patch16_224,
        'deit_tiny_patch16_224': deit_tiny_patch16_224,
        'deit_small_patch16_224': deit_small_patch16_224,
        'deit_tiny_distilled_patch16_224': deit_tiny_distilled_patch16_224,
        'deit_small_distilled_patch16_224': deit_small_distilled_patch16_224,
        'deit_base_patch16_384': deit_base_patch16_384,
        'deit_base_distilled_patch16_384': deit_base_distilled_patch16_384,
        'regnety_160': regnety_160,
        'midmap_deit_tiny_patch16_224': midmap_deit_tiny_patch16_224,
        'midmap_deit_tiny3_patch16_224': midmap_deit_tiny3_patch16_224,
        'midmap_deit_tiny6_patch16_224': midmap_deit_tiny6_patch16_224,
        'midmap_deit_tiny9_patch16_224': midmap_deit_tiny9_patch16_224,
        'midmap_deit_small_patch16_224': midmap_deit_small_patch16_224,
        'midmap_deit_small3_patch16_224': midmap_deit_small3_patch16_224,
        'midmap_deit_small6_patch16_224': midmap_deit_small6_patch16_224,
        'midmap_deit_small9_patch16_224': midmap_deit_small9_patch16_224,
        'midmap_deit_base_patch16_224': midmap_deit_base_patch16_224,
        'midmap_deit_base3_patch16_224': midmap_deit_base3_patch16_224,
        'midmap_deit_base6_patch16_224': midmap_deit_base6_patch16_224,
        'midmap_deit_base9_patch16_224': midmap_deit_base9_patch16_224,
        'deit_base6_patch16_224': deit_base6_patch16_224,
        'deit_small6_patch16_224': deit_small6_patch16_224,
        'deit_tiny6_patch16_224': deit_tiny6_patch16_224,
        'deit_base9_patch16_224': deit_base9_patch16_224,
        'deit_small9_patch16_224': deit_small9_patch16_224,
        'deit_tiny9_patch16_224': deit_tiny9_patch16_224,
    }
    return models[model_name](**kwargs)


def create_model(
        model_name,
        num_classes=1000,
        in_chans=3,
        checkpoint_path=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs):
    """Create model by name with given parameters"""
    model = get_model_by_name(
        model_name, num_classes=num_classes, in_chans=in_chans,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate, **kwargs
    )
    if checkpoint_path is not None:
        param_dict = ms.load_checkpoint(checkpoint_path)
        ms.load_param_into_net(model, param_dict)

    return model


def create_teacher_model(
        model_name,
        checkpoint_path=None,
        **kwargs):
    """Create model by name with given parameters"""

    model = get_model_by_name(
        model_name, **kwargs
    )
    if checkpoint_path is not None:
        param_dict = ms.load_checkpoint(checkpoint_path)
        ms.load_param_into_net(model, param_dict)
    return model

def create_LearngeneInstances(blk_length,init_mode, nb_classes, drop, drop_path):
    LearngeneInstances = []
    if blk_length == 6:
        deit_base6 = create_model(
            model_name='deit_base6_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        deit_small6 = create_model(
            model_name='deit_small6_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        deit_tiny6 = create_model(
            model_name='deit_tiny6_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        LearngeneInstances = [deit_tiny6, deit_small6, deit_base6]
    elif blk_length == 9:
        deit_base9 = create_model(
            model_name='deit_base9_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        deit_small9 = create_model(
            model_name='deit_small9_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        deit_tiny9 = create_model(
            'deit_tiny9_patch16_224',
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,)
        LearngeneInstances = [deit_tiny9, deit_small9, deit_base9]
    if init_mode == 'scratch':
        print('Learngene Instances are trained from scratch!')
        return LearngeneInstances
    LearngeneInstances = init_LearngenePool(LearngeneInstances, blk_length, init_mode)
    return LearngeneInstances

def create_learngene_pool(args):
    instances = create_LearngeneInstances(blk_length=args.blk_length, init_mode=args.init_learngenepool_mode,
                                     nb_classes=args.num_classes, drop=args.drop, drop_path=args.drop_path)
    model = LearngenePool(instances, blk_legth=args.blk_length, mode=args.init_stitch_mode)
    return model

def create_learngene_pool_eval(args):
    instances = create_LearngeneInstances(blk_length=args.blk_length, init_mode='scratch',
                                     nb_classes=args.num_classes, drop=args.drop, drop_path=args.drop_path)
    model = LearngenePool(instances, blk_legth=args.blk_length, mode='scratch')
    if args.blk_length==6:
        #ckpt_path = './results/24-04-29_040449_Learngenepool6_0/last_checkpoint/Learngenepool6.ckpt'
        ckpt_path = './results/24-05-15_113319_Learngenepool6_0/last_checkpoint/Learngenepool6.ckpt'
        
    elif args.blk_length==9:
        ckpt_path = './pool/Learngenepool9.ckpt'
    else:
        raise ValueError(f'blk_length={args.blk_length}is not viable')
    param_dict = ms.load_checkpoint(ckpt_path)
    print(param_dict)
    new_param_dict = {}
    for name, param in param_dict.items():
        new_name = name.replace('model.', '')
        new_name = new_name.replace('_network.', '')
        new_name = new_name.replace('stitch_layers', '0')
        new_name = new_name.replace('0.0','0')
        new_param_dict[new_name] = param
        print(new_name)
    print("begin load")
    load_param_into_net(model, new_param_dict)
    print(".............................................")
    for name,value in model.parameters_and_names():
        print(name)
    return model