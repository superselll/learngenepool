import numpy as np
import mindspore as ms
import mindspore.nn as nn
import random
from mindspore import Tensor
from mindspore.ops import combinations

def combination(x):
    res = []

    for i in range(len(x)):
        for j in range(i+1,len(x)):
            res.append((i,j))
    return res



def paired_stitching(depth=12, kernel_size=2, stride=1):
    blk_id = list(range(depth))
    i = 0
    stitch_cfgs = []
    stitch_id = -1
    stitching_layers_mappings = []

    while i < depth:
        ids = blk_id[i:i + kernel_size]
        has_new_stitches = False
        for j in ids:
            for k in ids:
                if (j, k) not in stitch_cfgs:
                    has_new_stitches = True
                    stitch_cfgs.append((j, k))
                    stitching_layers_mappings.append(stitch_id + 1)
        if has_new_stitches:
            stitch_id += 1
        i += stride
    num_stitches = stitch_id + 1
    return stitch_cfgs, stitching_layers_mappings, num_stitches

def get_stitch_configs(depth=6, kernel_size=2, stride=1, num_models=3, nearest_stitching=True):
    '''This function assumes the two model have the same depth, for demonstrating DeiT stitching.

    Args:
        depth: number of blocks in the model
        kernel_size: size of the stitching sliding window
        stride: stride of the stitching sliding window
        num_models: number of models to be stitched
        nearest_stitching: whether to use nearest stitching
    '''

    stitch_cfgs, layers_mappings, num_stitches = paired_stitching(depth, kernel_size, stride)
    # [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6), (6, 7), (7, 6), (7, 7),     (7, 8), (8, 7), (8, 8), (8, 9), (9, 8), (9, 9), (9, 10), (10, 9), (10, 10), (10, 11), (11, 10), (11, 11)]
    # [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]
    # 11

    # [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5)]
    # [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    # 5
    model_combinations = []
    candidates = Tensor(list(range(num_models)))  # [0, 1, 2]
    for i in range(1, num_models + 1):  # i = 1, 2, 3
        model_combinations += list(combinations(candidates, i).asnumpy().astype(dtype=int))  # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    if nearest_stitching:
        # remove tiny-base
        # model_combinations.pop(model_combinations.index((0, 2)))

        # remove three model settings
        model_combinations.pop(model_combinations.index((0, 1, 2)))

    total_configs = []

    for comb in model_combinations:  # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        if len(comb) == 1:
            total_configs.append({
                'comb_id': comb,
                'stitch_cfgs': [],
                'stitch_layers': []
            })
            continue

        for cfg, layer_mapping_id in zip(stitch_cfgs, layers_mappings):
            if len(comb) == 2:
                total_configs.append({
                    'comb_id': comb,
                    'stitch_cfgs': [cfg],
                    'stitch_layers': [layer_mapping_id]
                })
            else:
                last_out_id = cfg[1]
                for second_cfg, second_layer_mapping_id in zip(stitch_cfgs, layers_mappings):
                    middle_id, end_id = second_cfg
                    if middle_id < last_out_id:
                        continue
                    total_configs.append({
                        'comb_id': comb,
                        'stitch_cfgs': [cfg, second_cfg],
                        'stitch_layers': [layer_mapping_id, second_layer_mapping_id]
                    })
    return total_configs, num_stitches

class StitchingLayer(nn.Cell):
    def __init__(self, in_dim=None, out_dim=None):
        super().__init__()

        if in_dim == out_dim:
            self.transform = nn.Identity()
        elif in_dim != out_dim:
            self.transform = nn.Dense(in_dim, out_dim)

    def init_stitch_weights_bias(self, weight, bias=None):
        self.transform.weight.set_data(weight)
        if bias != None:
            self.transform.bias.set_data(bias)

    def construct(self, x):
        x = self.transform(x)
        return x




class LearngenePool(nn.Cell):
    '''
    Stitching from learngene Pool
    '''
    def __init__(self, learngene_instances, nearest_stitching=False, blk_legth=0, mode='ours', choose=0):
        super(LearngenePool, self).__init__()
        self.blk_length = blk_legth
        self.mode = mode
        self.choose = choose
        self.instances = nn.CellList(learngene_instances)
        self.instances_depths = [len(instances.blocks) for instances in self.instances]  # get the number of blocks of each learngene instance

        blk_stitch_cfgs, num_stitches = get_stitch_configs(self.instances_depths[0], kernel_size=2, stride=1,
                                                           num_models=len(self.instances), nearest_stitching=nearest_stitching)
        self.num_stitches = num_stitches

        #candidate_combinations = list(combinations(Tensor(list(range(len(learngene_instances)))), 2))  # [(0, 1), (0, 2), (1, 2)]
        candidate_combinations = combination(range(len(learngene_instances)))
        if nearest_stitching:
            candidate_combinations.pop(candidate_combinations.index((0, 2)))  # [(0, 1), (0, 2), (1, 2)]
        self.candidate_combinations = candidate_combinations  # [(0, 1), (0, 2), (1, 2)]

        self.stitch_layers = nn.CellList()
        self.stitching_map_id = {}
        for i, cand in enumerate(candidate_combinations):
            front, end = cand
            self.stitch_layers.append(
                nn.CellList([StitchingLayer(self.instances[front].embed_dim, self.instances[end].embed_dim) for _ in
                               range(num_stitches)])
            )
            self.stitching_map_id[f'{front}-{end}'] = i

        self.stitch_configs = {i: cfg for i, cfg in enumerate(blk_stitch_cfgs)}
        # logger.info(self.stitch_configs)
        self.num_configs = len(blk_stitch_cfgs)
        if self.blk_length == 6:
            self.stitch_cfg_ids = [0 ,2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # blk_length=6
        elif self.blk_length == 9:
            self.stitch_cfg_ids = [0, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                              49, 50, 51, 52]  # blk_length=9
        else:
            raise ValueError(f'blk_length={self.blk_length} is not viable')

        self.initialize_stitching_weights()

    def reset_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id

    def initialize_stitching_weights(self):
        if self.mode == 'scratch':
            return
        elif self.mode == 'ours':
            if self.blk_length == 6:
                ckpt_paths = [
                    './distill/deit_base_patch16_224-deit_tiny6_patch16_224/LearngenePool_[front,end]/TransLayerMap_checkpoint.ckpt',
                    './distill/deit_base_patch16_224-deit_small6_patch16_224/LearngenePool_[end]/TransLayerMap_checkpoint.ckpt',]
            elif  self.blk_length == 9:
                ckpt_paths = [
                    './distill/deit_base_patch16_224-deit_tiny9_patch16_224/LearngenePool_[front,end]/TransLayerMap_checkpoint.ckpt',
                    './distill/deit_base_patch16_224-deit_small9_patch16_224/LearngenePool_[end]/TransLayerMap_checkpoint.ckpt',]
            else:
                raise ValueError(f'blk_length = {self.blk_length} is not available.')
                return

            tiny_base_ckpt = ms.load_checkpoint(ckpt_paths[0])
            total_tiny_base_ckpt_weight = []
            #total_tiny_base_ckpt_bias = []
            for i in range(int(len(tiny_base_ckpt)/2)):
                total_tiny_base_ckpt_weight.append(tiny_base_ckpt['{}.transform.weight'.format(i)])
                #total_tiny_base_ckpt_bias.append(tiny_base_ckpt['trans_layer_map.{}.transform.bias'.format(i)])
            mean_tiny_base_ckpt_weight = (ms.ops.stack(total_tiny_base_ckpt_weight).mean(axis=0)).permute(1, 0)
            #mean_tiny_base_ckpt_bias = ms.ops.stack(total_tiny_base_ckpt_bias).mean(axis=0)
            for j in range(len(self.stitch_layers[1])):  # initialize stitching layers from 0-2
                self.stitch_layers[1][j].init_stitch_weights_bias(mean_tiny_base_ckpt_weight, None)

        else:
            raise ValueError(f'mode = {self.mode} is not available.')

    def construct(self, x):

        if self.training:
            stitch_cfg_id = random.choice(self.stitch_cfg_ids)
            # stitch_cfg_id = np.random.randint(0, self.num_configs)  # random sampling during training
        else:
            stitch_cfg_id = self.stitch_cfg_ids[self.choose]  #choose第一次默认为0
        # print(stitch_cfg_id)
        #{'comb_id': (0, 1, 2), 'stitch_cfgs': [(0, 0), (2, 3)], 'stitch_layers': [0, 2]}
        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']  # (0, 1)   从第0个辅助模型缝到第1个
        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']  # [(2, 3)]   第0个辅助模型从0开始从2结束，第一个辅助模型从3开始一直到结尾
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']  # [2]

        # 如果comb_id是1，那么表示不缝合，那就直接返回该网络的输出
        if len(comb_id) == 1:
            # simply forward the instance
            out = self.instances[int(comb_id[0])](x)
            return out
        else:
            # 如果comb_id!=1，表示有两个网络之间的缝合，首先对图片进行patch_embed操作，选择第一个instance的patch_embed
            x = self.instances[int(comb_id[0])].forward_patch_embed(x)

            front_id = 0
            for i in range(len(stitch_cfgs)):
                cfg = stitch_cfgs[i]
                #for i, cfg in enumerate(stitch_cfgs):  # [(0, 0), (2, 3)]
                end_id = int(cfg[0]) + 1
                # 经过第一个网络的前半部分
                for blk in self.instances[int(comb_id[i])].blocks[front_id:end_id]:
                    x = blk(x)

                # Stich layer的部分
                front_id = cfg[1]
                # 从第comb_id[i]个instance缝到第comb_id[i+1]
                key = str(comb_id[i]) + '-' + str(
                    comb_id[i + 1])  # 前面已经把要缝合的两个相邻层做成了id，比如0-1,id=0,{'0-1': 0, '0-2': 1, '1-2': 2}
                stitch_projection_id = self.stitching_map_id[key]  # 使用第几个缝合层
                sl_id = stitch_layer_ids[i]  # 使用该缝合层的第几个缝合块
                x = self.stitch_layers[int(stitch_projection_id)][int(sl_id)](x)  # 缝

            # 经过最后一个网络的后半部分
            for blk in self.instances[int(comb_id[-1])].blocks[front_id:]:
                x = blk(x)

            x = self.instances[int(comb_id[-1])].norm(x)
            x = self.instances[int(comb_id[-1])].head(x[:, 0])
            # x = nn.Softmax()(x)
            #x = self.instances[int(comb_id[-1])].head2(x)
            return x

