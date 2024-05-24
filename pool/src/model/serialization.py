# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Model and parameters serialization."""
from __future__ import absolute_import
from __future__ import division

import copy
import json
import os
import shutil
import stat
import threading
from threading import Thread, RLock
from collections import defaultdict, OrderedDict
from io import BytesIO

import math
import sys
import time
import numpy as np

from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model
from mindspore.train.print_pb2 import Print

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import log as logger
from mindspore._checkparam import check_input_data, check_input_dataset
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.common.api import _MindsporeFunctionExecutor
from mindspore.common.api import _get_parameter_layout
from mindspore.common.api import _generate_branch_control_input
from mindspore.common.initializer import initializer, One
from mindspore.common.parameter import Parameter, _offload_if_config
from mindspore.common.tensor import Tensor
from mindspore.common._utils import is_shape_unknown
from mindspore.communication.management import get_rank, get_group_size
from mindspore.experimental import MapParameter
from mindspore.parallel._cell_wrapper import get_allgather_cell
from mindspore.parallel._tensor import _load_tensor, _get_tensor_strategy, _get_tensor_slice_index
from mindspore.parallel._tensor import _reshape_param_data, _reshape_param_data_with_weight
from mindspore.parallel._utils import _infer_rank_list, _remove_repeated_slices, _is_in_auto_parallel_mode
from mindspore.parallel._parallel_serialization import _convert_to_list, _convert_to_layout, _build_searched_strategy, \
    _restore_group_info_list
from mindspore.parallel._ps_context import _set_checkpoint_load_status, _store_warm_up_ptr_by_tensor, \
    _store_warm_up_ptr_by_tensor_list, _cache_enable
from mindspore.train._utils import read_proto
from mindspore._c_expression import load_mindir, _encrypt, _decrypt, _is_cipher_file, dynamic_obfuscate_mindir, \
    split_mindir, split_dynamic_mindir
# from ..ops.operations._opaque_predicate_registry import add_opaque_predicate, clean_funcs

tensor_to_ms_type = {"Int8": mstype.int8, "UInt8": mstype.uint8, "Int16": mstype.int16, "UInt16": mstype.uint16,
                     "Int32": mstype.int32, "UInt32": mstype.uint32, "Int64": mstype.int64, "UInt64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_, "str": mstype.string}

tensor_to_np_type = {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
                     "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_, "str": "U"}

np_type_convert = {"int32": np.int32, "float32": np.float32, "float16": np.float16, "float64": np.float64}

mindir_to_tensor_type = {1: mstype.float32, 2: mstype.uint8, 3: mstype.int8, 4: mstype.uint16,
                         5: mstype.int16, 6: mstype.int32, 7: mstype.int64, 10: mstype.float16,
                         11: mstype.float64, 12: mstype.uint32, 13: mstype.uint64}

_ckpt_mutex = RLock()

# unit is KB
SLICE_SIZE = 512 * 1024
PROTO_LIMIT_SIZE = 1024 * 1024 * 2
TOTAL_SAVE = 1024 * 1024
PARAMETER_SPLIT_SIZE = 1024 * 1024 * 1024
ENCRYPT_BLOCK_SIZE = 64 * 1024
INT_64_MAX = 9223372036854775807


def _special_process_par(par, new_par):
    """
    Processes the special condition.

    Like (12,2048,1,1)->(12,2048), this case is caused by GE 4 dimensions tensor.
    """
    par_shape_len = len(par.data.shape)
    new_par_shape_len = len(new_par.data.shape)
    if new_par_shape_len <= par_shape_len:
        return False

    for i in range(new_par_shape_len - par_shape_len):
        if new_par.data.shape[par_shape_len + i] != 1:
            return False

    new_val = new_par.data.asnumpy()
    new_val = new_val.reshape(par.data.shape)
    par.set_data(Tensor(new_val, par.data.dtype))
    return True


def _update_param(param, new_param, strict_load):
    """Updates param's data from new_param's data."""
    if isinstance(param.data, Tensor) and isinstance(new_param.data, Tensor):
        if param.data.shape != new_param.data.shape:
            if not _special_process_par(param, new_param):
                logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
                msg = (f"For 'load_param_into_net', {param.name} in the argument 'net' should have the same shape "
                       f"as {param.name} in the argument 'parameter_dict'. But got its shape {param.data.shape} in"
                       f" the argument 'net' and shape {new_param.data.shape} in the argument 'parameter_dict'."
                       f"May you need to check whether the checkpoint you loaded is correct or the batch size and "
                       f"so on in the 'net' and 'parameter_dict' are same.")
                raise RuntimeError(msg)

        if param.data.dtype != new_param.data.dtype:
            if _type_convert(param, new_param, strict_load):
                new_tensor = Tensor(new_param.data.asnumpy(), param.data.dtype)
                param.set_data(new_tensor, param.sliced)
                return

            logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
            msg = (f"For 'load_param_into_net', {param.name} in the argument 'net' should have the same type as "
                   f"{param.name} in the argument 'parameter_dict'. but got its type {param.data.dtype} in the "
                   f"argument 'net' and type {new_param.data.dtype} in the argument 'parameter_dict'."
                   f"May you need to check whether the checkpoint you loaded is correct.")
            raise RuntimeError(msg)

        param.set_data(new_param.data, param.sliced)
        return

    if isinstance(param.data, Tensor) and not isinstance(new_param.data, Tensor):
        if param.data.shape != (1,) and param.data.shape != ():
            logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
            msg = (f"For 'load_param_into_net', {param.name} in the argument 'parameter_dict' is "
                   f"scalar, then the shape of {param.name} in the argument 'net' should be "
                   f"(1,) or (), but got shape {param.data.shape}."
                   f"May you need to check whether the checkpoint you loaded is correct.")
            raise RuntimeError(msg)
        param.set_data(initializer(new_param.data, param.data.shape, param.data.dtype))

    elif isinstance(new_param.data, Tensor) and not isinstance(param.data, Tensor):
        logger.critical("Failed to combine the net and the parameters for param %s.", param.name)
        msg = (f"For 'load_param_into_net', {param.name} in the argument 'parameter_dict' is Tensor, "
               f"then {param.name} in the argument 'net' also should be Tensor, but got {type(param.data)}."
               f"May you need to check whether the checkpoint you loaded is correct.")
        raise RuntimeError(msg)

    else:
        param.set_data(type(param.data)(new_param.data))


def _type_convert(param, new_param, strict_load):
    """Whether to convert parameter's type during load checkpoint into network."""
    float_type = (mstype.float16, mstype.float32, mstype.float64)
    int_type = (mstype.int8, mstype.int16, mstype.int32, mstype.int64)
    if not strict_load and ({param.data.dtype, new_param.data.dtype}.issubset(float_type) or
                            {param.data.dtype, new_param.data.dtype}.issubset(int_type)):
        logger.warning(f"The type of {new_param.name}:{new_param.data.dtype} in 'parameter_dict' is different from "
                       f"the type of it in 'net':{param.data.dtype}, then the type convert from "
                       f"{new_param.data.dtype} to {param.data.dtype} in the network.")
        return True
    return False


def _save_weight(checkpoint_dir, model_name, iteration, params):
    """Save model weight into checkpoint."""
    logger.debug(f"Checkpoint dir is: '{checkpoint_dir}'")
    exist_ckpt_file_list = []
    if os.path.exists(checkpoint_dir):
        for exist_ckpt_name in os.listdir(checkpoint_dir):
            file_prefix = os.path.join(model_name, "_iteration_")
            if exist_ckpt_name.startswith(file_prefix):
                exist_ckpt_file_list.append(exist_ckpt_name)

        param_dict = OrderedDict()
        for key in params.keys():
            value = params[key]
            weight_type = value[0]
            weight_shape = value[1]
            weight_data = value[2]
            weight_size = value[3]
            weight_np = np.array(weight_data, dtype=weight_type.lower())
            logger.debug(f"weight_type: '{weight_type}', weight_shape: '{weight_shape}', weight_size: "
                         f"'{weight_size}', weight_np.nbytes: '{weight_np.nbytes}'")

            param_dict[key] = [weight_shape, weight_type, weight_np]
        ckpt_file_save_name = model_name + "_iteration_" + iteration + ".ckpt"
        ckpt_file_save_path = os.path.join(checkpoint_dir, ckpt_file_save_name)

        _exec_save(ckpt_file_save_path, param_dict)

        for exist_ckpt_name in exist_ckpt_file_list:
            os.remove(os.path.join(checkpoint_dir, exist_ckpt_name))
        logger.info(f"Save weight to checkpoint file path '{ckpt_file_save_path}' success.")
    else:
        logger.warning(f"Checkpoint dir: '{checkpoint_dir}' is not existed.")


def _exec_save(ckpt_file_name, data_list, enc_key=None, enc_mode="AES-GCM", map_param_inc=False):
    """Execute the process of saving checkpoint into file."""
    try:
        with _ckpt_mutex:
            if os.path.exists(ckpt_file_name):
                os.chmod(ckpt_file_name, stat.S_IWUSR)
                os.remove(ckpt_file_name)
            with open(ckpt_file_name, "ab") as f:
                plain_data = None
                if enc_key is not None:
                    plain_data = BytesIO()

                for name, value in data_list.items():
                    if name == "random_op":
                        _write_random_seed(name, value, f)
                        continue
                    if value[0] == "mapparameter":
                        _write_mapparameter(name, value, f, map_param_inc)
                        continue
                    if value[0] == "offload_parameter":
                        new_value = value[1:]
                        new_value[2] = value[3].asnumpy().reshape(-1)
                        _write_parameter_data(name, new_value, f, enc_key, plain_data)
                        _offload_if_config(value[3])
                        continue
                    if isinstance(value[2], Tensor):
                        _write_hugeparameter(name, value, f)
                        continue

                    _write_parameter_data(name, value, f, enc_key, plain_data)

                if enc_key is not None:
                    plain_data.seek(0)
                    max_block_size = ENCRYPT_BLOCK_SIZE * 1024
                    block_data = plain_data.read(max_block_size)
                    while block_data:
                        f.write(_encrypt(block_data, len(block_data), enc_key, len(enc_key), enc_mode))
                        block_data = plain_data.read(max_block_size)

                os.chmod(ckpt_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical("Failed to save the checkpoint file %s. Maybe don't have the permission to write files, "
                        "or the disk space is insufficient and so on.", ckpt_file_name)
        raise e


def _write_random_seed(name, value, f):
    """Write random op into protobuf file."""
    checkpoint_list = Checkpoint()
    param_value = checkpoint_list.value.add()
    param_value.tag = name
    param_tensor = param_value.tensor
    param_tensor.dims.extend(0)
    param_tensor.tensor_type = "random_op"
    param_tensor.tensor_content = value
    f.write(checkpoint_list.SerializeToString())


def _write_parameter_data(name, value, f, enc_key, plain_data):
    """Write parameter data into protobuf file."""
    data_size = value[2].nbytes / 1024
    if data_size > SLICE_SIZE:
        slice_count = math.ceil(data_size / SLICE_SIZE)
        param_slice_list = np.array_split(value[2], slice_count)
    else:
        param_slice_list = [value[2]]

    for param_slice in param_slice_list:
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        param_tensor = param_value.tensor
        param_tensor.dims.extend(value[0])
        param_tensor.tensor_type = value[1]
        param_tensor.tensor_content = param_slice.tobytes()

        if enc_key is None:
            f.write(checkpoint_list.SerializeToString())
        else:
            plain_data.write(checkpoint_list.SerializeToString())


def _write_mapparameter(name, value, f, map_param_inc=False):
    """Write map parameter into protobuf file."""
    while True:
        logger.info("Checkpoint save map_parameter.")
        data_map_slice = value[1].export_slice_data(map_param_inc)
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        map_tensor = param_value.maptensor
        for numpy_data in data_map_slice[:3]:
            tensor_pro = map_tensor.tensor.add()
            tensor_pro.dims.extend(numpy_data.shape)
            tensor_pro.tensor_type = str(numpy_data.dtype)
            tensor_pro.tensor_content = numpy_data.reshape(-1).tobytes()
        f.write(checkpoint_list.SerializeToString())
        if data_map_slice[3]:
            break


def _write_hugeparameter(name, value, f):
    """Write huge parameter into protobuf file."""
    slice_num = value[2].slice_num
    offset = 0
    max_size = value[0][0]
    for param_slice in range(slice_num):
        checkpoint_list = Checkpoint()
        param_value = checkpoint_list.value.add()
        param_value.tag = name
        param_tensor = param_value.tensor
        param_tensor.dims.extend(value[0])
        param_tensor.tensor_type = value[1]
        param_key = value[3]
        numpy_data = value[2].asnumpy_of_slice_persistent_data(param_key, param_slice)
        if offset + numpy_data.shape[0] > max_size:
            numpy_data = numpy_data[:max_size - offset]
        param_tensor.tensor_content = numpy_data.tobytes()
        f.write(checkpoint_list.SerializeToString())
        offset += numpy_data.shape[0]


def _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name):
    """Check save_obj and ckpt_file_name for save_checkpoint."""
    if not isinstance(save_obj, (nn.Cell, list, dict)):
        raise TypeError("For 'save_checkpoint', the parameter 'save_obj' must be nn.Cell, list or dict, "
                        "but got {}.".format(type(save_obj)))
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'save_checkpoint', the parameter {} for checkpoint file name is invalid,"
                        "'ckpt_file_name' must be "
                        "string, but got {}.".format(ckpt_file_name, type(ckpt_file_name)))
    ckpt_file_name = os.path.abspath(ckpt_file_name)
    if os.path.isdir(ckpt_file_name):
        raise IsADirectoryError("For 'save_checkpoint', the parameter `ckpt_file_name`: {} is a directory, "
                                "it must be a file name.".format(ckpt_file_name))
    if not ckpt_file_name.endswith('.ckpt'):
        ckpt_file_name += ".ckpt"
    return ckpt_file_name


def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM", choice_func=None, **kwargs):
    r"""
    Save checkpoint to a specified file.

    Args:
        save_obj (Union[Cell, list, dict]): The object to be saved. The data type can be :class:`mindspore.nn.Cell`,
            list, or dict. If a list, it can be the returned value of `Cell.trainable_params()`, or a list of dict
            elements(each element is a dictionary, like [{"name": param_name, "data": param_data},...], the type of
            `param_name` must be string, and the type of `param_data` must be parameter or Tensor); If dict,
            it can be the returned value of `mindspore.load_checkpoint()`.
        ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: ``True`` .
        async_save (bool): Whether to open an independent thread to save the checkpoint file. Default: ``False`` .
        append_dict (dict): Additional information that needs to be saved. The key of dict must be str, the value
                            of dict must be one of int, float, bool, string, Parameter or Tensor. Default: ``None`` .
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is ``None`` , the encryption
                                      is not required. Default: ``None`` .
        enc_mode (str): This parameter is valid only when enc_key is not set to ``None`` . Specifies the encryption
                        mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"`` and ``"SM4-CBC"`` .
                        Default: ``"AES-GCM"`` .
        choice_func (function) : A function for saving custom selected parameters. The input value of `choice_func` is
                                 a parameter name in string type, and the returned value is a bool.
                                 If returns ``True`` , the Parameter that matching the custom condition will be saved.
                                 If returns ``False`` , the Parameter that not matching the custom condition will not
                                 be saved. Default: ``None`` .
        kwargs (dict): Configuration options dictionary.

    Raises:
        TypeError: If the parameter `save_obj` is not :class:`mindspore.nn.Cell` , list or dict type.
        TypeError: If the parameter `integrated_save` or `async_save` is not bool type.
        TypeError: If the parameter `ckpt_file_name` is not string type.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> ms.save_checkpoint(net, "./lenet.ckpt",
        ...                    choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
        >>> param_dict1 = ms.load_checkpoint("./lenet.ckpt")
        >>> print(param_dict1)
        {'conv2.weight': Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)}
        >>> params_list = net.trainable_params()
        >>> ms.save_checkpoint(params_list, "./lenet_list.ckpt",
        ...                    choice_func=lambda x: x.startswith("conv") and not x.startswith("conv2"))
        >>> param_dict2 = ms.load_checkpoint("./lenet_list.ckpt")
        >>> print(param_dict2)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}
        >>> ms.save_checkpoint(param_dict2, "./lenet_dict.ckpt")
        >>> param_dict3 = ms.load_checkpoint("./lenet_dict.ckpt")
        >>> print(param_dict3)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/r2.2/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    ckpt_file_name = _check_save_obj_and_ckpt_file_name(save_obj, ckpt_file_name)
    integrated_save = Validator.check_bool(integrated_save)
    async_save = Validator.check_bool(async_save)
    append_dict = _check_append_dict(append_dict)
    enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
    enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)
    map_param_inc = kwargs.get('incremental', False)
    logger.info("Execute the process of saving checkpoint files.")

    save_obj = _convert_save_obj_to_param_list(save_obj, integrated_save, append_dict, choice_func)

    if append_dict:
        append_info_list = []
        for k_name, value in append_dict.items():
            if not isinstance(value, str):
                value = Tensor(value)
            append_info_list.append({"name": k_name, "data": value})
        save_obj.extend(append_info_list)

    data_list = OrderedDict()
    with _ckpt_mutex:
        for param in save_obj:
            if param["name"] == "random_op":
                data_list["random_op"] = param["data"]
                continue
            key = param["name"]
            data_list[key] = []
            if isinstance(param["data"], MapParameter):
                data_list[param["name"]].append("mapparameter")
                data_list[param["name"]].append(param["data"])
                continue
            if isinstance(param["data"], list):
                if param["data"][0] == "persistent_data":
                    _save_param_list_data(data_list, key, param)
                elif param["data"][0] == "offload_parameter":
                    data_list[key].append("offload_parameter")
                    _save_param_list_data(data_list, key, param)

            if isinstance(param["data"], str):
                data_list[key].append([0])
                data_list[key].append('str')
                data = np.array(param["data"])
                data_list[key].append(data)
            else:
                if isinstance(param["data"], Parameter):
                    param["data"].init_data()
                dims = []
                if param['data'].shape == ():
                    dims.append(0)
                else:
                    for dim in param['data'].shape:
                        dims.append(dim)
                data_list[key].append(dims)
                tensor_type = str(param["data"].dtype)
                data_list[key].append(tensor_type)
                data = param["data"].asnumpy().reshape(-1)
                data_list[key].append(data)

    if async_save:
        data_copy = copy.deepcopy(data_list)
        thr = Thread(target=_exec_save, args=(ckpt_file_name, data_copy, enc_key, enc_mode), name="asyn_save_ckpt")
        thr.start()
    else:
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode, map_param_inc)

    logger.info("Saving checkpoint process is finished.")


def _convert_list_to_param_list(save_obj, choice_func):
    """Convert a list of Parameter to param_list."""
    param_list = []
    if not save_obj:
        return param_list
    if isinstance(save_obj[0], dict):
        param_list = [param for param in save_obj if choice_func is None or choice_func(param["name"])]
    else:
        for param in save_obj:
            if isinstance(param, Parameter):
                if choice_func is not None and not choice_func(param.name):
                    continue
                each_param = {"name": param.name, "data": param}
                param_list.append(each_param)
            else:
                raise TypeError(f"For save_checkpoint, when save_obj is made up by list of Parameter,"
                                f"the param should be parameter, but got {type(param)}")
    return param_list


def _convert_dict_to_param_dict(save_obj, choice_func):
    """Convert a dict of Parameter to param_list."""
    param_list = []
    for (key, value) in save_obj.items():
        if isinstance(key, str) and isinstance(value, (Parameter, str)):
            if choice_func is not None and not choice_func(key):
                continue
            each_param = {"name": key, "data": value}
            param_list.append(each_param)
        else:
            raise TypeError(f"For save_checkpoint, when save_obj is made up by dict, the key should be str and"
                            f"value should be Parameter, but got the type of key is {type(key)} and"
                            f"the type of value is {type(value)}")
    return param_list


def _convert_cell_param_and_names_to_dict(save_obj, choice_func):
    """Convert cell.parameters_and_names to OrderedDict."""
    param_dict = OrderedDict()
    for _, param in save_obj.parameters_and_names():
        not_sliced = not param.sliced
        is_graph_mode = context.get_context('mode') == context.GRAPH_MODE
        # All parameters are initialized immediately under PyNative mode, skip this judgement.
        judgment = not_sliced or param.has_init
        if is_graph_mode and _is_in_auto_parallel_mode() and judgment:
            continue
        if choice_func is not None and not choice_func(param.name):
            continue
        # Add suffix for cache_enabled parameter, and then parameter can carry key info.
        # Notice that suffix needs be removed when loading into net.
        if param.cache_enable:
            param_dict[param.name + ".__param_key__" + str(param.key)] = param
        else:
            param_dict[param.name] = param
    return param_dict


def _convert_cell_to_param_list(save_obj, integrated_save, append_dict, choice_func):
    """Convert nn.Cell to param_list."""
    param_list = []
    parameter_layout_dict = save_obj.parameter_layout_dict
    if _is_in_auto_parallel_mode() and not parameter_layout_dict:
        parameter_layout_dict = _get_parameter_layout()
    if not _is_in_auto_parallel_mode():
        save_obj.init_parameters_data()
    param_dict = _convert_cell_param_and_names_to_dict(save_obj, choice_func)
    if append_dict and "random_op" in append_dict:
        phase = 'train' + '.' + str(save_obj.create_time) + '.' + str(id(save_obj)) + '.' + save_obj.arguments_key
        if phase in save_obj.compile_cache and _executor.has_compiled(phase):
            random_byte = _executor._graph_executor.get_random_status(phase)
            param_list.append({"name": "random_op", "data": random_byte})
        append_dict.pop("random_op")
    for (key, value) in param_dict.items():
        each_param = {"name": key}
        if isinstance(value, MapParameter):
            each_param["data"] = value
            param_list.append(each_param)
            continue

        if value.data.is_persistent_data():
            # list save persistent_data: [Tensor, shape, type, param.key]
            param_data = ["persistent_data", value.data, value.param_info.origin_shape, str(value.dtype), value.key]
        elif value.data.offload_file_path() != "":
            # list save offload data: [Param, shape, type, param.key]
            param_data = ["offload_parameter"]
            param_tensor = value.data
            if key in parameter_layout_dict:
                param_tensor = _get_merged_param_data(save_obj, parameter_layout_dict, key, param_tensor,
                                                      integrated_save)
            param_data.append(param_tensor)
            param_data.append(param_tensor.shape)
            param_data.append(str(param_tensor.dtype))
            param_data.append(value.key)
        else:
            param_data = Tensor(value.data.asnumpy())

            # in automatic model parallel scenario, some parameters were split to all the devices,
            # which should be combined before saving
            if key in parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, parameter_layout_dict, key, param_data,
                                                    integrated_save)

        each_param["data"] = param_data
        param_list.append(each_param)
    return param_list


def _convert_save_obj_to_param_list(save_obj, integrated_save, append_dict, choice_func):
    """Convert a save_obj to param_list."""
    if isinstance(save_obj, list):
        return _convert_list_to_param_list(save_obj, choice_func)

    if isinstance(save_obj, dict):
        return _convert_dict_to_param_dict(save_obj, choice_func)

    return _convert_cell_to_param_list(save_obj, integrated_save, append_dict, choice_func)


def _save_param_list_data(data_list, key, param):
    """Save persistent data into save_obj."""
    dims = []
    # persistent_data shape can not be ()
    for dim in param['data'][2]:
        dims.append(dim)
    data_list[key].append(dims)
    data_list[key].append(param['data'][3])
    data_list[key].append(param['data'][1])
    data_list[key].append(param['data'][4])


def _check_append_dict(append_dict):
    """Check the argument append_dict for save_checkpoint."""
    if append_dict is None:
        return append_dict
    if not isinstance(append_dict, dict):
        raise TypeError("For 'save_checkpoint', the argument 'append_dict' must be dict, but got "
                        "{}.".format(type(append_dict)))
    for key, value in append_dict.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, bool, str, Parameter, Tensor)):
            raise TypeError(f"For 'save_checkpoint', the type of dict 'append_info' must be key: string, "
                            f"value: int, float or bool, but got key: {type(key)}, value: {type(value)}")
    return append_dict


def _check_load_obfuscate(**kwargs):
    if 'obf_func' in kwargs.keys():
        customized_func = _check_customized_func(kwargs.get('obf_func'))
        clean_funcs()
        add_opaque_predicate(customized_func.__name__, customized_func)
        return True
    return False


def load(file_name, **kwargs):
    """
    Load MindIR.

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.

        kwargs (dict): Configuration options dictionary.

            - dec_key (bytes): Byte-type key used for decryption. The valid length is 16, 24, or 32.
            - dec_mode (Union[str, function]): Specifies the decryption mode, to take effect when dec_key is set.

              - Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC' or customized decryption. Default: 'AES-GCM'.
              - For details of using the customized decryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/r2.0/model_encrypt_protection.html>`_.

            - obf_func (function): A python function used for loading obfuscated MindIR model, which can refer to
              `obfuscate_model()
              <https://www.mindspore.cn/docs/en/r2.2/api_python/mindspore/mindspore.obfuscate_model.html>`_.

    Returns:
        GraphCell, a compiled graph that can executed by `GraphCell`.

    Raises:
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to parse MindIR file.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>>
        >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
        >>> input_tensor = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
        >>> graph = ms.load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input_tensor)
        >>> print(output)
        [[[[4. 6. 4.]
           [6. 9. 6.]
           [4. 6. 4.]]]]

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading MindIR
          <https://mindspore.cn/tutorials/en/r2.2/beginner/save_load.html#saving-and-loading-mindir>`_
    """
    if not isinstance(file_name, str):
        raise ValueError("For 'load', the argument 'file_name' must be string, but "
                         "got {}.".format(type(file_name)))
    if not file_name.endswith(".mindir"):
        raise ValueError("For 'load', the argument 'file_name'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_name'.")
    if not os.path.exists(file_name):
        raise ValueError("For 'load', the argument 'file_name'(MindIR file) does not exist, "
                         "please check whether the 'file_name' is correct.")
    file_name = os.path.abspath(file_name)

    # set customized functions for dynamic obfuscation
    obfuscated = _check_load_obfuscate(**kwargs)

    logger.info("Execute the process of loading mindir.")
    if 'dec_key' in kwargs.keys():
        dec_key = Validator.check_isinstance('dec_key', kwargs.get('dec_key'), bytes)
        dec_mode = "AES-GCM"
        dec_func = None
        if 'dec_mode' in kwargs.keys():
            if callable(kwargs.get('dec_mode')):
                dec_mode = "Customized"
                dec_func = kwargs.get('dec_mode')
            else:
                dec_mode = Validator.check_isinstance('dec_mode', kwargs.get('dec_mode'), str)
        graph = load_mindir(file_name, dec_key=dec_key, key_len=len(dec_key), dec_mode=dec_mode,
                            decrypt=dec_func, obfuscated=obfuscated)
    else:
        graph = load_mindir(file_name, obfuscated=obfuscated)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Load MindIR failed. The file may be encrypted and decrypt failed, you "
                               "can check whether the values of the arguments 'dec_key' and 'dec_mode'"
                               " are the same as when exported MindIR file, or check the file integrity.")
        raise RuntimeError("Load MindIR failed.")
    return graph


def export_split_mindir(file_name, device_num=8, rank_id=0, dynamic=True, sapp=False):
    """
    Auto Split MindIR.

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.
        device_num (int): device number.
        rank_id (int): rank id.
        dynamic (bool): Indicates whether the model is a dynamic shape mindir model.
        sapp (bool): Indicates whether to automatically generate split strategy through SAPP.

    Raises:
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to split MindIR file.

    Examples:
        >>> import mindspore as ms
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>>
        >>> ms.export_split_mindir("net.mindir", device_num=8, rank_id=0)

    """
    if not isinstance(file_name, str):
        raise ValueError("For 'Split MindIR', the argument 'file_name' must be string, but "
                         "got {}.".format(type(file_name)))
    if not file_name.endswith(".mindir"):
        raise ValueError("For 'Split MindIR', the argument 'file_name'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_name'.")
    if not os.path.exists(file_name):
        raise ValueError("For 'Split MindIR', the argument 'file_name'(MindIR file) does not exist, "
                         "please check whether the 'file_name' is correct.")
    file_name = os.path.abspath(file_name)

    logger.info("Execute the process of export and split mindir.")
    dynamic = True
    if dynamic:
        graph = split_dynamic_mindir(file_name, device_num, rank_id, sapp)
    else:
        graph = split_mindir(file_name)

    if graph is None:
        if _is_cipher_file(file_name):
            raise RuntimeError("Export and split MindIR failed. The file may be encrypted and decrypt failed, you "
                               "can check whether the values of the arguments 'dec_key' and 'dec_mode'"
                               " are the same as when exported MindIR file, or check the file integrity.")
        raise RuntimeError("Export and split MindIR failed.")
    return graph


def _check_param_type(param_config, key, target_type, requested):
    """check type of parameters"""
    if key in param_config:
        if not isinstance(param_config[key], target_type):
            raise TypeError("The type of {} must be {}, but got {}.".format(key, target_type, type(param_config[key])))
        if key == 'obf_random_seed':
            if param_config[key] > INT_64_MAX or param_config[key] <= 0:
                raise ValueError(
                    "'obf_random_seed' must be in (0, INT_64_MAX({})], but got {}.".format(INT_64_MAX,
                                                                                           param_config[key]))
        return param_config[key]
    if requested:
        raise ValueError("The parameter {} is requested, but not got.".format(key))
    if key == "obf_random_seed":
        return 0
    return None


def _check_customized_func(customized_func):
    """ check customized function of dynamic obfuscation """
    if not callable(customized_func):
        raise TypeError(
            "'customized_func' must be a function, but not got {}.".format(type(customized_func)))
    # test customized_func
    try:
        func_result = customized_func(1.0, 1.0)
    except Exception as ex:
        raise TypeError("customized_func must be a function with two inputs, but got exception: {}".format(ex))
    else:
        if not isinstance(func_result, bool):
            raise TypeError("Return value of customized_func must be boolean, but got: {}".format(type(func_result)))
    return customized_func


def _check_obfuscate_params(obf_config):
    """Check obfuscation parameters, including obf_random_seed, obf_ratio, customized_func"""
    if 'obf_random_seed' not in obf_config.keys() and 'customized_func' not in obf_config.keys():
        raise ValueError(
            "At least one of 'obf_random_seed' or 'customized_func' must be set in obf_config, but got None of them.")
    obfuscate_type = _check_param_type(obf_config, "type", str, False)
    if obfuscate_type not in (None, "dynamic"):
        raise ValueError("Only 'dynamic' type is supported by now, but got {}.".format(obfuscate_type))
    if ('obf_ratio' in obf_config) and isinstance(obf_config['obf_ratio'], str):
        if obf_config['obf_ratio'] not in ["small", "medium", "large"]:
            raise ValueError("'obf_ratio' can only be 'small', 'medium', 'large' or float, but got {}.".format(
                obf_config['obf_ratio']))
        ratio_dict = {"small": 0.1, "medium": 0.3, "large": 0.6}
        obf_config['obf_ratio'] = ratio_dict.get(obf_config['obf_ratio'])
    obf_ratio = _check_param_type(obf_config, "obf_ratio", float, True)
    if (obf_ratio <= 0) or (obf_ratio > 1):
        raise ValueError("'obf_ratio' must be in (0, 1] if it is a float, but got {}.".format(obf_config['obf_ratio']))
    customized_funcs = []
    if 'customized_func' in obf_config.keys():
        device_target = context.get_context('device_target')
        if device_target in ["GPU", "Ascend"]:
            raise ValueError(
                "Customized func mode only support 'device_target'='CPU, but got {}.".format(device_target))
        customized_funcs.append(_check_customized_func(obf_config['customized_func']))
    obf_random_seed = _check_param_type(obf_config, "obf_random_seed", int, False)
    return obf_ratio, customized_funcs, obf_random_seed


def obfuscate_model(obf_config, **kwargs):
    """
    Obfuscate a model of MindIR format. Obfuscation means changing the struct of a network without affecting its
    predict correctness. The obfuscated model can prevent attackers from stealing the model.

    Args:
        obf_config (dict): obfuscation config.

            - type (str): The type of obfuscation, only 'dynamic' is supported until now.
            - original_model_path (str): The path of MindIR format model that need to be obfuscated. If the original
              model is encrypted, then enc_key and enc_mode should be provided.
            - save_model_path (str): The path to save the obfuscated model.
            - model_inputs (list(Tensor)): The inputs of the original model, the values of Tensor can be random, which
              is the same as using :func:`mindspore.export`.
            - obf_ratio (Union(float, str)): The ratio of nodes in original model that would be obfuscated. `obf_ratio`
              should be in range of (0, 1] or in ["small", "medium", "large"]. "small", "medium" and "large" are
              correspond to 0.1, 0.3, and 0.6 respectively.
            - customized_func (function): A python function used for customized function mode, which used for control
              the switch branch of obfuscation structure. The outputs of customized_func should be boolean and const (
              Reference to 'my_func()' in
              `tutorials <https://www.mindspore.cn/mindarmour/docs/en/r2.0/dynamic_obfuscation_protection.html>`_).
              This function needs to ensure that its result is constant for any input. Users can refer to opaque
              predicates. If customized_func is set, then it should be passed to :func:`mindspore.load` interface
              when loading obfuscated model.
            - obf_random_seed (int): Obfuscation random seed, which should be in (0, 9223372036854775807]. The
              structure of obfuscated models corresponding to different random seeds is different. If
              `obf_random_seed` is set, then it should be passed to :class:`nn.GraphCell()` interface when loading
              obfuscated model. It should be noted that at least one of `customized_func` or `obf_random_seed` should
              be set, and the latter mode would be applied if both of them are set.

        kwargs (dict): Configuration options dictionary.

            - enc_key (bytes): Byte type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (str): Specifies the encryption mode, to take effect when dec_key is set.
              Option: 'AES-GCM' | 'AES-CBC' | 'SM4-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: If `obf_config` is not a dict.
        ValueError: If `enc_key` is passed and `enc_mode` is not in ["AES-GCM", "AES-CBC", "SM4-CBC"].
        ValueError: If `original_model_path` is not provided in `obf_config`.
        ValueError: If the model saved in `original_model_path` has been obfuscated.
        ValueError: If `save_model_path` is not provided in `obf_config`.
        ValueError: If `obf_ratio` is not provided in `obf_config`.
        ValueError: If both `customized_func` and `obf_random_seed` are not provided in `obf_config`.
        ValueError: If `obf_random_seed` is not in (0, 9223372036854775807].
        ValueError: If `original_model_path` is not exist or `original_model_path` is not end with '.mindir'.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> obf_config = {'original_model_path': "./net.mindir",
        ...          'save_model_path': "./obf_net",
        ...          'model_inputs': [input1, ],
        ...          'obf_ratio': 0.1, 'obf_random_seed': 173262358423}
        >>> ms.obfuscate_model(obf_config)
        >>> obf_func = ms.load("obf_net.mindir")
        >>> obf_net = nn.GraphCell(obf_func, obf_random_seed=173262358423)
        >>> print(obf_net(input1).asnumpy())
    """
    if not isinstance(obf_config, dict):
        raise TypeError("'obf_config' must be a dict, but got {}.".format(type(obf_config)))
    file_path = _check_param_type(obf_config, "original_model_path", str, True)
    if not file_path.endswith(".mindir"):
        raise ValueError("For 'obfuscate_model', the argument 'file_path'(MindIR file) should end with '.mindir', "
                         "please input the correct 'file_path'.")
    if not os.path.exists(file_path):
        raise ValueError("For 'obfuscate_model', the argument 'file_path'(MindIR file) does not exist, "
                         "please check whether the 'file_path' is correct.")
    saved_path = _check_param_type(obf_config, "save_model_path", str, True)
    model_inputs = _check_param_type(obf_config, "model_inputs", list, True)
    for item in model_inputs:
        if not isinstance(item, Tensor):
            raise TypeError("The item in 'model_inputs' must be Tensor, but got {}.".format(type(item)))
        if -1 in item.shape:
            raise ValueError(
                "Dynamic shape input is not supported now, but got the shape of inputs: {}.".format(item.shape))
    obf_ratio, customized_funcs, obf_random_seed = _check_obfuscate_params(obf_config)
    if customized_funcs and obf_random_seed > 0:
        logger.warning("Although 'customized_func' and 'obf_random_seed' are set, the 'obf_random_seed' mode would be"
                       " applied, remember to set 'obf_random_seed' when loading obfuscated model.")

    if obf_random_seed == 0:  # apply customized_func mode
        clean_funcs()
        for func in customized_funcs:
            add_opaque_predicate(func.__name__, func)
        branch_control_input = 0
    else:  # apply password mode
        branch_control_input = _generate_branch_control_input(obf_random_seed)

    if 'enc_key' in kwargs.keys():
        enc_key = Validator.check_isinstance('enc_key', kwargs.get('enc_key'), bytes)
        enc_mode = "AES-GCM"
        if 'enc_mode' in kwargs.keys():
            enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)
            if enc_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
                raise ValueError(
                    "Only MindIR files that encrypted with 'AES-GCM', 'AES-CBC' or 'SM4-CBC' is supported for"
                    "obfuscate_model(), but got {}.".format(enc_mode))
        obf_graph = dynamic_obfuscate_mindir(file_name=file_path, obf_ratio=obf_ratio,
                                             branch_control_input=branch_control_input, dec_key=enc_key,
                                             key_len=len(enc_key),
                                             dec_mode=enc_mode)
    else:
        obf_graph = dynamic_obfuscate_mindir(file_name=file_path, obf_ratio=obf_ratio,
                                             branch_control_input=branch_control_input)

    obf_net = nn.GraphCell(obf_graph)
    if obf_random_seed != 0:
        append_y_tensor = Tensor(np.ones((1, 1)).astype(np.int32))
        model_inputs += [append_y_tensor,]
    export(obf_net, *model_inputs, file_name=saved_path, file_format="MINDIR", **kwargs)


def load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None,
                    dec_key=None, dec_mode="AES-GCM", specify_prefix=None, choice_func=None):
    """
    Load checkpoint info from a specified file.

    Note:
        - `specify_prefix` and `filter_prefix` do not affect each other.
        - If none of the parameters are loaded from checkpoint file, it will throw ValueError.
        - `specify_prefix` and `filter_prefix` are in the process of being deprecated,
          `choice_func` is recommended instead.
          And using either of those two args will override `choice_func` at the same time.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        net (Cell): The network where the parameters will be loaded. Default: ``None`` .
        strict_load (bool): Whether to strict load the parameter into net. If ``False`` , it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        filter_prefix (Union[str, list[str], tuple[str]]): Deprecated(see `choice_func`). Parameters starting with the
            filter_prefix will not be loaded. Default: ``None`` .
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is ``None`` , the decryption
                                      is not required. Default: ``None`` .
        dec_mode (str): This parameter is valid only when dec_key is not set to ``None`` . Specifies the decryption
                        mode, currently supports ``"AES-GCM"`` and ``"AES-CBC"`` and ``"SM4-CBC"`` .
                        Default: ``"AES-GCM"`` .
        specify_prefix (Union[str, list[str], tuple[str]]): Deprecated(see `choice_func`). Parameters starting with the
            specify_prefix will be loaded. Default: ``None`` .
        choice_func (Union[None, function]) : Input value of the function is a Parameter name of type string,
            and the return value is a bool. If returns ``True`` , the Parameter
            that matches the custom condition will be loaded. If returns ``False`` , the Parameter that
            matches the custom condition will be removed. Default: ``None`` .

    Returns:
        Dict, key is parameter name, value is a Parameter or string. When the `append_dict` parameter of
        :func:`mindspore.save_checkpoint` and the `append_info` parameter of :class:`mindspore.train.CheckpointConfig`
        are used to save the checkpoint, `append_dict` and `append_info` are dict types, and their value are string,
        then the return value obtained by loading checkpoint is string, and in other cases the return value is
        Parameter.

    Raises:
        ValueError: Checkpoint file's format is incorrect.
        ValueError: Parameter's dict is None after load checkpoint file.
        TypeError: The type of `specify_prefix` or `filter_prefix` is incorrect.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name,
        ...                                 choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
        >>> print(param_dict["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
        >>> def func(param_name):
        ...     whether_load = False
        ...     if param_name.startswith("conv"):
        ...         whether_load = True
        ...     if param_name.startswith("conv1"):
        ...         whether_load = False
        ...     return whether_load
        >>> param_dict1 = ms.load_checkpoint(ckpt_file_name, choice_func=func)
        >>> print(param_dict1["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
        >>> def func(param_name):
        ...     whether_load = False
        ...     if param_name.startswith("conv1"):
        ...         whether_load = True
        ...     return whether_load
        >>> param_dict2 = ms.load_checkpoint(ckpt_file_name, choice_func=func)
        >>> print(param_dict2)
        {'conv1.weight': Parameter (name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)}

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/r2.2/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    ckpt_file_name = _check_ckpt_file_name(ckpt_file_name)
    specify_prefix = _check_prefix(specify_prefix)
    filter_prefix = _check_prefix(filter_prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode)

    parameter_dict = {}
    try:
        param_data_list = []
        map_data_list = [[], [], []]
        map_shape_list = [0, 0, 0]
        if specify_prefix:
            logger.warning("For load_checkpoint, this parameter `specity_prefix` will be deprecated, "
                           "please use `choice_func` instead.")
        if filter_prefix:
            logger.warning("For load_checkpoint, this parameter `filter_prefix` will be deprecated, "
                           "please use `choice_func` instead.")
        for element_id, element in enumerate(checkpoint_list.value):
            if element.tag == "random_op":
                parameter_dict["random_op"] = element.tensor.tensor_content
                continue
            if not _whether_load_param(specify_prefix, filter_prefix, element.tag):
                continue
            if specify_prefix is None and filter_prefix is None and \
                    choice_func is not None and not choice_func(element.tag):
                continue
            if element.tensor.ByteSize() == 0:
                _load_map_parameter(checkpoint_list, element, element_id, map_data_list, map_shape_list, parameter_dict)
                if element.tag in parameter_dict:
                    map_data_list = [[], [], []]
                    map_shape_list = [0, 0, 0]
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type.get(data_type)
            ms_type = tensor_to_ms_type[data_type]
            if data_type == 'str':
                str_length = int(len(data) / 4)
                np_type = np_type + str(str_length)
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                new_data = b"".join(param_data_list)
                param_data = np.frombuffer(new_data, np_type)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0] and data_type == 'str':
                    parameter_dict[element.tag] = str(element_data[0])
                else:
                    if dims == [0] and 'Float' in data_type:
                        param_data = float(param_data[0])
                    if dims == [0] and 'Int' in data_type:
                        param_data = int(param_data[0])
                    if dims not in ([0], [1]):
                        param_data = param_data.reshape(list(dims))
                    parameter = Parameter(Tensor(param_data, ms_type), name=element.tag)
                    parameter_dict[element.tag] = parameter
                    _offload_if_config(parameter)

        logger.info("Loading checkpoint files process is finished.")

    except BaseException as e:
        logger.critical("Failed to load the checkpoint file '%s'.", ckpt_file_name)
        raise ValueError(e.__str__() + "\nFor 'load_checkpoint', "
                                       "failed to load the checkpoint file {}.".format(ckpt_file_name)) from e

    if not parameter_dict:
        raise ValueError(f"The loaded parameter dict is empty after filter or specify, please check whether "
                         f"'filter_prefix' or 'specify_prefix' are set correctly.")

    if _warm_up_host_cache_enabled(parameter_dict):
        (is_worker, net_dict, warm_up_dict) = _warm_up_host_cache(parameter_dict, net)
    if net is not None:
        load_param_into_net(net, parameter_dict, strict_load)
    if _warm_up_host_cache_enabled(parameter_dict):
        _warm_up_host_cache_post_process(is_worker, net_dict, warm_up_dict)

    return parameter_dict


def _load_map_parameter(checkpoint_list, element, element_id, map_data_list,
                        map_shape_list, parameter_dict):
    """load map parameter."""
    logger.info("Checkpoint load map_parameter.")
    if (element_id != len(checkpoint_list.value) - 1) and \
            element.tag == checkpoint_list.value[element_id + 1].tag:
        for index, tensor in enumerate(element.maptensor.tensor):
            data = tensor.tensor_content
            data_type = tensor.tensor_type
            np_type = np_type_convert.get(data_type)
            element_data = np.frombuffer(data, np_type)
            map_data_list[index].append(element_data)
            map_shape_list[index] += tensor.dims[0]
    else:
        map_array = []
        for index, tensor in enumerate(element.maptensor.tensor):
            data = tensor.tensor_content
            data_type = tensor.tensor_type
            np_type = np_type_convert.get(data_type)
            element_data = np.frombuffer(data, np_type)
            map_data_list[index].append(element_data)
            new_data = b"".join(map_data_list[index])
            param_data = np.frombuffer(new_data, np_type)
            dims = tensor.dims
            dims[0] += map_shape_list[index]
            param_data = param_data.reshape(list(dims))
            map_array.append(param_data)
        parameter_dict[element.tag] = map_array


def _check_ckpt_file_name(ckpt_file_name):
    """Check function load_checkpoint's ckpt_file_name."""
    if not isinstance(ckpt_file_name, str):
        raise TypeError("For 'load_checkpoint', the argument 'ckpt_file_name' must be string, "
                        "but got {}.".format(type(ckpt_file_name)))

    if ckpt_file_name[-5:] != ".ckpt":
        raise ValueError("For 'load_checkpoint', the checkpoint file should end with '.ckpt', please "
                         "input the correct 'ckpt_file_name'.")

    ckpt_file_name = os.path.abspath(ckpt_file_name)
    if not os.path.exists(ckpt_file_name):
        raise ValueError("For 'load_checkpoint', the checkpoint file: {} does not exist, please check "
                         "whether the 'ckpt_file_name' is correct.".format(ckpt_file_name))

    return ckpt_file_name


def _check_prefix(prefix):
    """Check the correctness of the parameters."""
    if prefix is None:
        return prefix
    if not isinstance(prefix, (str, list, tuple)):
        raise TypeError("For 'load_checkpoint', the type of 'specify_prefix' or 'filter_prefix' must be string, "
                        "list[string] or tuple[string], but got {}.".format(str(type(prefix))))
    if isinstance(prefix, str):
        prefix = (prefix,)
    if not prefix:
        raise ValueError("For 'load_checkpoint', the argument 'specify_prefix' or 'filter_prefix' can't be empty when"
                         " 'specify_prefix' or 'filter_prefix' is list or tuple.")
    for index, pre in enumerate(prefix):
        if not isinstance(pre, str):
            raise TypeError("For 'load_checkpoint', when 'specify_prefix' or 'filter_prefix' is list or tuple, "
                            "the element in it must be string, but got "
                            f"{str(type(pre))} at index {index}.")
        if pre == "":
            raise ValueError("For 'load_checkpoint', the value of 'specify_prefix' or 'filter_prefix' "
                             "can't include ''.")
    return prefix


def _parse_ckpt_proto(ckpt_file_name, dec_key, dec_mode):
    """Parse checkpoint protobuf."""
    checkpoint_list = Checkpoint()
    try:
        if dec_key is None:
            with open(ckpt_file_name, "rb") as f:
                pb_content = f.read()
        else:
            pb_content = _decrypt(ckpt_file_name, dec_key, len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError("For 'load_checkpoint', failed to decrypt the checkpoint file.")
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        if _is_cipher_file(ckpt_file_name):
            err_info = "Failed to read the checkpoint file {}. The file may be encrypted or tempered with, " \
                       "please pass in the correct 'dec_key' or check the file integrity.".format(ckpt_file_name)
        else:
            err_info = "Failed to read the checkpoint file {}. May not have permission to read it, please check" \
                       " the correct of the file.".format(ckpt_file_name)
        logger.error(err_info)
        raise ValueError(err_info) from e
    return checkpoint_list


def _whether_load_param(specify_prefix, filter_prefix, param_name):
    """Checks whether the load the parameter after `specify_prefix` or `filter_prefix`."""
    whether_load = True
    if specify_prefix:
        whether_load = False
        for prefix in specify_prefix:
            if param_name.startswith(prefix):
                whether_load = True
                break
    if filter_prefix:
        for prefix in filter_prefix:
            if param_name.startswith(prefix):
                whether_load = False
                break
    return whether_load


def _init_parameter_data_in_parallel_mode(net, parameter_dict):
    """In parallel mode, only init the paraemters in ckpt."""
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict and param.has_init:
            logger.warning("{} is not init while load ckpt.".format(param.name))
            new_tensor = param.init_data()
            param._update_tensor_data(new_tensor)


def load_param_into_net(net, parameter_dict, strict_load=False):
    """
    Load parameters into network, return parameter list that are not loaded in the network.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file,
                               it is a dictionary consisting of key: parameters's name, value: parameter.
        strict_load (bool): Whether to strict load the parameter into net. If ``False`` , it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .

    Returns:
        param_not_load (List), the parameter name in model which are not loaded into the network.
        ckpt_not_load (List), the parameter name in checkpoint file which are not loaded into the network.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load, _ = ms.load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading the Model Weight
          <https://mindspore.cn/tutorials/en/r2.2/beginner/save_load.html#saving-and-loading-the-model-weight>`_
    """
    if not isinstance(net, nn.Cell):
        logger.critical("Failed to combine the net and the parameters.")
        msg = ("For 'load_param_into_net', the argument 'net' should be a Cell, but got {}.".format(type(net)))
        raise TypeError(msg)
    if not isinstance(parameter_dict, dict):
        logger.critical("Failed to combine the net and the parameters.")
        msg = ("For 'load_param_into_net', the argument 'parameter_dict' should be a dict, "
               "but got {}.".format(type(parameter_dict)))
        raise TypeError(msg)
    if "random_op" in parameter_dict.keys():
        net._add_attr("random_op_snapshot", parameter_dict["random_op"])
        parameter_dict.pop("random_op")
    for key, value in parameter_dict.items():
        if not isinstance(key, str) or not isinstance(value, (Parameter, str, list)):
            logger.critical("Load parameters into net failed.")
            msg = ("For 'parameter_dict', the element in the argument 'parameter_dict' should be a "
                   "'str' and 'Parameter' , but got {} and {}.".format(type(key), type(value)))
            raise TypeError(msg)

    strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    if not _is_in_auto_parallel_mode():
        net.init_parameters_data()
    else:
        _init_parameter_data_in_parallel_mode(net, parameter_dict)
    param_not_load = []
    ckpt_not_load = list(parameter_dict.keys())
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            if isinstance(param, MapParameter):
                param.import_data(parameter_dict[param.name])
                continue
            # Add has attr protection when load server checkpoint file on worker.
            if not hasattr(parameter_dict[param.name], "data"):
                continue
            new_param = copy.deepcopy(parameter_dict[param.name])
            _update_param(param, new_param, strict_load)
            ckpt_not_load.remove(param.name)
        else:
            param_not_load.append(param.name)
            print(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)

    logger.debug("Params not matched(in net but not in parameter_dict):")
    for param_name in param_not_load:
        logger.debug("%s", param_name)

    logger.info("Loading parameters into net is finished.")
    if param_not_load:
        logger.warning("For 'load_param_into_net', "
                       "{} parameters in the 'net' are not loaded, because they are not in the "
                       "'parameter_dict', please check whether the network structure is consistent "
                       "when training and loading checkpoint.".format(len(param_not_load)))
        for param_name in param_not_load:
            logger.warning("{} is not loaded.".format(param_name))
    return param_not_load, ckpt_not_load


def _warm_up_host_cache_enabled(parameter_dict):
    """Warm up host cache enabled."""
    if _cache_enable():
        return True
    for key in parameter_dict.keys():
        if key.find(".__param_key__") != -1:
            return True
    return False


def _warm_up_host_cache(parameter_dict, net):
    """Warm up host cache."""
    ms_role = os.getenv("MS_ROLE")
    is_worker = ms_role == "MS_WORKER"
    param_key_dict = {}
    # Traverse key, value in parameter_dict, warm up param key and record param key into param_key_dict.
    if is_worker:
        net.init_parameters_data()
        net_dict = {}
        for name, value in net.parameters_and_names():
            net_dict[name] = value
        for param_name, value in parameter_dict.items():
            pos = param_name.find(".__param_key__")
            if pos != -1:
                net_param_name = param_name[:pos]
                param_key_dict[param_name] = net_param_name
                net_value = None
                if net_param_name not in net_dict:
                    logger.warning("net param name : %s is not in net", net_param_name)
                else:
                    net_value = net_dict.get(net_param_name, None)
                pos += len(".__param_key__")
                param_key = int(param_name[pos:])
                value_is_map_parameter = isinstance(value, list) and len(value) == 3
                if value_is_map_parameter and (net_value is None or isinstance(net_value, Parameter)):
                    key_tensor = Tensor.from_numpy(value[0])
                    value_tensor = Tensor.from_numpy(value[1])
                    status_tensor = Tensor.from_numpy(value[2])
                    _store_warm_up_ptr_by_tensor_list(param_key, key_tensor, value_tensor, status_tensor)
                elif not isinstance(value, list) and isinstance(net_value, Parameter):
                    _store_warm_up_ptr_by_tensor(param_key, value)
                else:
                    logger.warning("Unknown matches parameter type %s and net_value %s", type(value), type(net_value))
    else:
        for param_name, value in parameter_dict.items():
            pos = param_name.find(".__param_key__")
            if pos != -1:
                net_param_name = param_name[:pos]
                param_key_dict[param_name] = net_param_name
    # Split param key from parameter_dict since worker cannot load param key.
    warm_up_dict = {}
    for key, value in param_key_dict.items():
        if is_worker:
            warm_up_dict[value] = parameter_dict.pop(key)
        else:
            parameter_dict[value] = parameter_dict.pop(key)
    return (is_worker, parameter_dict, warm_up_dict)


def _warm_up_host_cache_post_process(is_worker, net_dict, warm_up_dict):
    """Warm up host cache post process."""
    if is_worker:
        net_dict.update(warm_up_dict)
    _set_checkpoint_load_status(True)


def _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load):
    """When some net parameter did not load, try to continue loading."""
    prefix_name = ""
    longest_name = param_not_load[0]
    while prefix_name != longest_name and param_not_load:
        logger.debug("Count: {} parameters has not been loaded, try to continue loading.".format(len(param_not_load)))
        prefix_name = longest_name
        for net_param_name in param_not_load:
            for dict_name in parameter_dict:
                if dict_name.endswith(net_param_name):
                    prefix_name = dict_name[:-len(net_param_name)]
                    break
            if prefix_name != longest_name:
                break

        if prefix_name != longest_name:
            logger.warning(f"For 'load_param_into_net', remove parameter prefix name: {prefix_name},"
                           f" continue to load.")
            for _, param in net.parameters_and_names():
                new_param_name = prefix_name + param.name
                if param.name in param_not_load and new_param_name in parameter_dict:
                    new_param = parameter_dict[new_param_name]
                    _update_param(param, new_param, strict_load)
                    param_not_load.remove(param.name)


def _save_graph(network, file_name):
    """
    Saves the graph of network to a file.

    Args:
        network (Cell): Obtain a pipeline through network for saving graph.
        file_name (str): Graph file name into which the graph will be saved.
    """
    logger.info("Execute the process of saving graph.")

    file_name = os.path.abspath(file_name)
    graph_pb = network.get_func_graph_proto()
    if graph_pb:
        with open(file_name, "wb") as f:
            os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
            f.write(graph_pb)


def _get_merged_param_data(net, parameter_layout_dict, param_name, param_data, integrated_save):
    """
    Gets the merged data(tensor) from tensor slice, by device arrangement and tensor map.

    Args:
        net (Cell): MindSpore network.
        param_name (str): The parameter name, which to be combined.
        param_data (Tensor): The parameter data on the local device, which was a slice of the whole parameter data.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene.
    Returns:
        Tensor, the combined tensor which with the whole data value.
    """
    layout = parameter_layout_dict[param_name]
    if len(layout) < 6:
        logger.info("The layout dict does not contain the key %s", param_name)
        return param_data

    dev_mat = layout[0]
    tensor_map = layout[1]
    uniform_split = layout[4]
    opt_shard_group = layout[5]

    allgather_net = None
    mp_weight = False
    for dim in tensor_map:
        if dim != -1:
            mp_weight = True
            break
    if param_name in net.parallel_parameter_merge_net_dict:
        allgather_net = net.parallel_parameter_merge_net_dict[param_name]
    else:
        logger.info("Need to create allgather net for %s", param_name)
        if integrated_save:
            if context.get_auto_parallel_context("pipeline_stages") > 1:
                raise RuntimeError("Pipeline Parallel don't support Integrated save checkpoint now.")
            if uniform_split == 0:
                raise RuntimeError("For 'save_checkpoint' and in automatic model parallel scene, when set "
                                   "'integrated_save' to True, the checkpoint will be integrated save, it "
                                   "is only supports uniform split tensor now.")
            # while any dim is not equal to -1, means param is split and needs to be merged
            # pipeline parallel need to be supported here later
            if mp_weight:
                allgather_net = get_allgather_cell(opt_shard_group, bool(opt_shard_group))
                object.__setattr__(allgather_net, "keep_input_unchanged", True)
            elif opt_shard_group:
                allgather_net = get_allgather_cell(opt_shard_group, False)
        elif opt_shard_group and context.get_auto_parallel_context("optimizer_weight_shard_aggregated_save"):
            allgather_net = get_allgather_cell(opt_shard_group, False)
        net.parallel_parameter_merge_net_dict[param_name] = allgather_net
    if allgather_net:
        param_data = allgather_net(param_data)
    if mp_weight and integrated_save:
        param_data = _reshape_param_data(param_data, dev_mat, tensor_map)
    return param_data


def export(net, *inputs, file_name, file_format, **kwargs):
    """
    Export the MindSpore network into an offline model in the specified format.

    Note:
        1. When exporting AIR, ONNX format, the size of a single tensor can not exceed 2GB.
        2. When file_name does not have a suffix, the system will automatically add one according to the file_format.
        3. Exporting functions decorated with :func:`mindspore.jit` to mindir format is supported.
        4. When exporting a function decorated with :func:`mindspore.jit`, the function should not involve
           class properties in calculations.

    Args:
        net (Union[Cell, function]): MindSpore network.
        inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.
        file_name (str): File name of the model to be exported.
        file_format (str): MindSpore currently supports 'AIR', 'ONNX' and 'MINDIR' format for exported model.

            - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
            - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
              for MindSpore models.

        kwargs (dict): Configuration options dictionary.

            - enc_key (byte): Byte-type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (Union[str, function]): Specifies the encryption mode, to take effect when enc_key is set.

              - For 'AIR' and 'ONNX' models, only customized encryption is supported.
              - For 'MINDIR', all options are supported. Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC'
                or Customized encryption.
                Default: 'AES-GCM'.
              - For details of using the customized encryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/r2.0/model_encrypt_protection.html>`_.

            - dataset (Dataset): Specifies the preprocessing method of the dataset, which is used to import the
              preprocessing of the dataset into MindIR.

            - obf_config (dict): obfuscation config.

              - type (str): The type of obfuscation, only 'dynamic' is supported until now.
              - obf_ratio (float, str): The ratio of nodes in original model that would be obfuscated. `obf_ratio`
                should be in range of (0, 1] or in ["small", "medium", "large"]. "small", "medium" and "large" are
                correspond to 0.1, 0.3, and 0.6 respectively.
              - customized_func (function): A python function used for customized function mode, which used for control
                the switch branch of obfuscation structure. The outputs of customized_func should be boolean and const (
                Reference to 'my_func()' in
                `tutorials <https://www.mindspore.cn/mindarmour/docs/en/r2.0/dynamic_obfuscation_protection.html>`_).
                This function needs to ensure that its result is constant for any input. Users can refer to opaque
                predicates. If customized_func is set, then it should be passed to `load()` interface when loading
                obfuscated model.
              - obf_random_seed (int): Obfuscation random seed, which should be in (0, 9223372036854775807]. The
                structure of obfuscated models corresponding to different random seeds is different. If
                `obf_random_seed` is set, then it should be passed to :class:`nn.GraphCell()` interface when loading
                obfuscated model. It should be noted that at least one of `customized_func` or `obf_random_seed` should
                be set, and the latter mode would be applied if both of them are set.

            - incremental (bool): export MindIR incrementally.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.export(net, input_tensor, file_name='lenet', file_format='MINDIR')

    Tutorial Examples:
        - `Saving and Loading the Model - Saving and Loading MindIR
          <https://mindspore.cn/tutorials/en/r2.2/beginner/save_load.html#saving-and-loading-mindir>`_
    """
    old_ms_jit_value = context.get_context("jit_syntax_level")
    context.set_context(jit_syntax_level=mindspore.STRICT)

    supported_formats = ['AIR', 'ONNX', 'MINDIR']
    if file_format not in supported_formats:
        raise ValueError(f"For 'export', 'file_format' must be one of {supported_formats}, but got {file_format}.")
    Validator.check_file_name_by_regular(file_name)
    logger.info("exporting model file:%s format:%s.", file_name, file_format)

    if check_input_dataset(*inputs, dataset_type=mindspore.dataset.Dataset):
        if len(inputs) != 1:
            raise RuntimeError(f"You can only serialize one dataset into MindIR, got " + str(len(inputs)) + " datasets")
        shapes, types, columns = inputs[0].output_shapes(), inputs[0].output_types(), inputs[0].get_col_names()
        kwargs['dataset'] = inputs[0]
        only_support_col = "image"

        inputs_col = list()
        for c, s, t in zip(columns, shapes, types):
            if only_support_col != c:
                continue
            inputs_col.append(Tensor(np.random.uniform(-1.0, 1.0, size=s).astype(t)))
        if not inputs_col:
            raise RuntimeError(f"Only supports parse \"image\" column from dataset now, given dataset has columns: "
                               + str(columns))
        inputs = tuple(inputs_col)

    file_name = os.path.abspath(file_name)
    if 'enc_key' in kwargs.keys():
        kwargs['enc_key'], kwargs['enc_mode'] = _check_key_mode_type(file_format, **kwargs)
    _export(net, file_name, file_format, *inputs, **kwargs)

    context.set_context(jit_syntax_level=old_ms_jit_value)


def _get_funcgraph(net, *inputs):
    """
    Compile the MindSpore network and get FuncGraph.

    Arg:
        net (Union[Cell, function]): MindSpore network.
        inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.

    Returns:
        FuncGraph, a mindspore._c_expression.FuncGraph obj.

    Raises:
        ValueError: input `net` is not a nn.Cell.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
        >>> ms.get_funcgraph(net, input_tensor)

    """
    if not isinstance(net, nn.Cell):
        raise ValueError(f"For get_funcgraph's parameter 'net', currently only support Cell right now.")
    phase_name = "lite_infer_predict" if _is_in_auto_parallel_mode() else "lite_infer_get_func_graph"
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    # pylint: disable=protected-access
    func_graph = _executor._get_func_graph(net, graph_id)
    return func_graph


def _export(net, file_name, file_format, *inputs, **kwargs):
    """
    It is an internal conversion function. Export the MindSpore prediction model to a file in the specified format.
    """
    logger.info("exporting model file:%s format:%s.", file_name, file_format)
    if "obf_config" in kwargs and file_format != "MINDIR":
        raise ValueError(f"Dynamic obfuscation only support for MindIR format, but got {file_format} format.")
    if file_format == 'AIR':
        _save_air(net, file_name, *inputs, **kwargs)
    elif file_format == 'ONNX':
        _save_onnx(net, file_name, *inputs, **kwargs)
    elif file_format == 'MINDIR':
        _save_mindir(net, file_name, *inputs, **kwargs)


def _check_key_mode_type(file_format, **kwargs):
    """check enc_key and enc_mode are valid"""
    enc_key = Validator.check_isinstance('enc_key', kwargs.get('enc_key'), bytes)
    enc_mode = kwargs.get('enc_mode')

    if callable(enc_mode):
        return enc_key, enc_mode

    enc_mode = 'AES-GCM'
    if 'enc_mode' in kwargs.keys():
        enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)

    if file_format in ('AIR', 'ONNX'):
        raise ValueError(f"AIR/ONNX only support customized encryption, but got {enc_mode}.")

    if enc_mode in ('AES-CBC', 'AES-GCM', 'SM4-CBC'):
        return enc_key, enc_mode
    raise ValueError(f"MindIR only support AES-GCM/AES-CBC/SM4-CBC encryption, but got {enc_mode}")


def _save_air(net, file_name, *inputs, **kwargs):
    """Save AIR format file."""
    phase_name = 'export.air'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name)
    if not file_name.endswith('.air'):
        file_name += ".air"
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    if "/" in file_name:
        real_path = os.path.abspath(file_name[:file_name.rfind("/")])
        os.makedirs(real_path, exist_ok=True)
    if 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys():
        _executor.export(file_name, graph_id, enc_key=kwargs.get('enc_key'), encrypt_func=kwargs.get('enc_mode'))
    else:
        _executor.export(file_name, graph_id)
    os.chmod(file_name, stat.S_IRUSR)


def _save_onnx(net, file_name, *inputs, **kwargs):
    """Save ONNX format file."""
    # When dumping ONNX file, switch network mode to infer when it is training(NOTE: ONNX only designed for prediction)
    if not isinstance(net, nn.Cell):
        raise ValueError(f"Export ONNX format model only support nn.Cell object, but got {type(net)}.")
    _check_dynamic_input(inputs)
    cell_mode = net.training
    net.set_train(mode=False)
    total_size = _calculation_net_size(net)
    if total_size > PROTO_LIMIT_SIZE:
        raise RuntimeError('Export onnx model failed. Network size is: {}G, it exceeded the protobuf: {}G limit.'
                           .format(total_size / 1024 / 1024, PROTO_LIMIT_SIZE / 1024 / 1024))
    phase_name = 'export.onnx'
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    onnx_stream = _executor._get_func_graph_proto(net, graph_id)
    if 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys():
        enc_mode = kwargs.get('enc_mode')
        onnx_stream = enc_mode(onnx_stream, kwargs.get('enc_key'))
    if not file_name.endswith('.onnx'):
        file_name += ".onnx"
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    with open(file_name, 'wb') as f:
        f.write(onnx_stream)
        os.chmod(file_name, stat.S_IRUSR)
    net.set_train(mode=cell_mode)


def _check_dynamic_input(inputs):
    for ele in inputs:
        if isinstance(ele, Tensor) and -1 in ele.shape:
            raise ValueError(f"Export ONNX format model not support dynamic shape mode.")


def _generate_front_info_for_param_data_file(is_encrypt, kwargs):
    front_info = bytes()
    check_code = sys.byteorder == "little"
    front_info += check_code.to_bytes(1, byteorder=sys.byteorder)
    front_info += bytes(63)
    if is_encrypt():
        front_info = _encrypt(front_info, len(front_info), kwargs.get('enc_key'),
                              len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
    return front_info


def _change_file(f, dirname, external_local, is_encrypt, kwargs):
    """Change to another file to write parameter data."""
    # The parameter has been not written in the file
    front_info = _generate_front_info_for_param_data_file(is_encrypt, kwargs)
    f.seek(0, 0)
    f.write(front_info)
    f.close()
    ori_data_file_name = f.name
    os.chmod(ori_data_file_name, stat.S_IRUSR)
    if os.path.getsize(ori_data_file_name) == 64:
        raise RuntimeError("The parameter size is exceed 1T,cannot export to the file")
    data_file_name = os.path.join(dirname, external_local)
    return _get_data_file(is_encrypt, kwargs, data_file_name)


def _get_data_file(is_encrypt, kwargs, data_file_name):
    """Get Data File to write parameter data."""
    # Reserves 64 bytes as spare information such as check data
    offset = 64
    if os.path.exists(data_file_name):
        os.chmod(data_file_name, stat.S_IWUSR)

    place_holder_data = bytes(offset)
    if is_encrypt():
        place_holder_data = _encrypt(place_holder_data, len(place_holder_data), kwargs["enc_key"],
                                     len(kwargs["enc_key"]), kwargs["enc_mode"])
    parameter_size = (offset / 1024)
    try:
        f = open(data_file_name, "wb")
        f.write(place_holder_data)
    except IOError:
        f.close()

    return f, parameter_size, offset


def _encrypt_data(is_encrypt, write_data, kwargs):
    """Encrypt parameter data."""
    if is_encrypt():
        if callable(kwargs.get('enc_mode')):
            enc_func = kwargs.get('enc_mode')
            write_data = enc_func(write_data, kwargs.get('enc_key'))
        else:
            write_data = _encrypt(write_data, len(write_data), kwargs.get('enc_key'),
                                  len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
    return write_data


def _split_save(net_dict, model, file_name, is_encrypt, **kwargs):
    """The function to save parameter data."""
    logger.warning("Parameters in the net capacity exceeds 1G, save MindIR model and parameters separately.")
    # save parameter
    if model.graph.map_parameter:
        raise ValueError("MapParameter not support save in split MindIR file now.")
    file_prefix = file_name.split("/")[-1]
    if file_prefix.endswith(".mindir"):
        file_prefix = file_prefix[:-7]
    current_path = os.path.abspath(file_name)
    dirname = os.path.dirname(current_path)
    data_path = os.path.join(dirname, file_prefix + "_variables")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path, exist_ok=True)
    os.chmod(data_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    index = 0
    external_local = os.path.join(file_prefix + "_variables", "data_" + str(index))
    data_file_name = os.path.join(dirname, external_local)
    f, parameter_size, offset = _get_data_file(is_encrypt, kwargs, data_file_name)
    try:
        for param_proto in model.graph.parameter:
            name = param_proto.name[param_proto.name.find(":") + 1:]
            param = net_dict[name]
            raw_data = param.data.get_bytes()
            data_length = len(raw_data)
            append_size = 0
            if data_length % 64 != 0:
                append_size = 64 - (data_length % 64)
                parameter_size += ((append_size + data_length) / 1024)
            if parameter_size > PARAMETER_SPLIT_SIZE:
                index += 1
                external_local = os.path.join(file_prefix + "_variables", "data_" + str(index))
                f, parameter_size, offset = _change_file(f, dirname, external_local, is_encrypt, kwargs)
                parameter_size += ((append_size + data_length) / 1024)
            param_proto.external_data.location = external_local
            param_proto.external_data.length = data_length
            param_proto.external_data.offset = offset
            write_data = raw_data + bytes(append_size)
            offset += (data_length + append_size)
            write_data = _encrypt_data(is_encrypt, write_data, kwargs)
            f.write(write_data)

        graph_file_name = os.path.join(dirname, file_prefix + "_graph.mindir")
        if os.path.exists(graph_file_name):
            os.chmod(graph_file_name, stat.S_IWUSR)
        with open(graph_file_name, 'wb') as model_file:
            os.chmod(graph_file_name, stat.S_IRUSR | stat.S_IWUSR)
            model_string = model.SerializeToString()
            if is_encrypt():
                model_string = _encrypt(model_string, len(model_string), kwargs.get('enc_key'),
                                        len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
            model_file.write(model_string)
            os.chmod(graph_file_name, stat.S_IRUSR)

        front_info = _generate_front_info_for_param_data_file(is_encrypt, kwargs)
        f.seek(0, 0)
        f.write(front_info)
    finally:
        f.close()
        os.chmod(data_file_name, stat.S_IRUSR)


def _msfunc_info(net, *inputs):
    """Get mindir stream and parameter dict of ms_function"""
    # pylint: disable=protected-access
    net_dict = OrderedDict()
    _ms_func_executor = _MindsporeFunctionExecutor(net, time.time() * 1e9)
    graph_id = _ms_func_executor.compile(net.__name__, *inputs)
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir')
    params = _ms_func_executor._graph_executor.get_params(graph_id)
    for name, value in params.items():
        net_dict[name] = Parameter(value, name=name)
    return mindir_stream, net_dict


def _cell_info(net, incremental, *inputs):
    """Get mindir stream and net dict of cell"""
    phase_name = "export.mindir"
    graph_id, _ = _executor.compile(net, *inputs, phase=phase_name, do_convert=False)
    # pylint: disable=protected-access
    mindir_stream = _executor._get_func_graph_proto(net, graph_id, 'mind_ir', incremental=incremental)
    # clean obfuscation config to prevent the next call
    _executor.obfuscate_config = None

    net_dict = net.parameters_dict()
    return mindir_stream, net_dict


def _set_obfuscate_config(**kwargs):
    """Set obfuscation config for executor."""
    logger.warning("Obfuscate model.")
    if 'enc_mode' in kwargs.keys():
        enc_mode = Validator.check_isinstance('enc_mode', kwargs.get('enc_mode'), str)
        if enc_mode not in ["AES-GCM", "AES-CBC", "SM4-CBC"]:
            raise ValueError(
                "Only MindIR files that encrypted with 'AES-GCM', 'AES-CBC' or 'SM4-CBC' is supported for"
                "obfuscation, but got {}.".format(enc_mode))
    obf_ratio, customized_funcs, obf_random_seed = _check_obfuscate_params(kwargs.get('obf_config'))
    if customized_funcs and obf_random_seed > 0:
        logger.warning("Although 'customized_func' and 'obf_random_seed' are set, the 'obf_random_seed' mode would be"
                       " applied, remember to set 'obf_random_seed' when loading obfuscated model.")

    if obf_random_seed == 0:  # apply customized_func mode
        device_target = context.get_context('device_target')
        if device_target in ["GPU", "Ascend"]:
            raise ValueError(
                "Customized func mode only support 'device_target'='CPU, but got {}.".format(device_target))
        clean_funcs()
        for func in customized_funcs:
            add_opaque_predicate(func.__name__, func)
    _executor.obfuscate_config = {'obf_ratio': obf_ratio, 'obf_random_seed': obf_random_seed}


def _save_mindir(net, file_name, *inputs, **kwargs):
    """Save MindIR format file."""
    # set obfuscate configs
    if 'obf_config' in kwargs.keys():
        _set_obfuscate_config(**kwargs)
        for item in inputs:
            if -1 in item.shape:
                raise ValueError(
                    "Dynamic shape input is not supported now, but got the shape of inputs: {}.".format(item.shape))

    incremental = kwargs.get('incremental', False)

    model = mindir_model()
    if not isinstance(net, nn.Cell):
        mindir_stream, net_dict = _msfunc_info(net, *inputs)
    else:
        mindir_stream, net_dict = _cell_info(net, incremental, *inputs)
    model.ParseFromString(mindir_stream)

    if kwargs.get('dataset'):
        check_input_data(kwargs.get('dataset'), data_class=mindspore.dataset.Dataset)
        dataset = kwargs.get('dataset')
        _save_dataset_to_mindir(model, dataset)

    save_together = _save_together(net_dict, model)
    is_encrypt = lambda: 'enc_key' in kwargs.keys() and 'enc_mode' in kwargs.keys()
    if save_together:
        _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs)
    else:
        _split_save(net_dict, model, file_name, is_encrypt, **kwargs)


def _save_mindir_together(net_dict, model, file_name, is_encrypt, **kwargs):
    """Save graph and parameter together."""
    for param_proto in model.graph.parameter:
        param_name = param_proto.name[param_proto.name.find(":") + 1:]
        if param_name in net_dict.keys():
            param_data = net_dict[param_name].data.get_bytes()
            param_proto.raw_data = param_data
        else:
            raise ValueError("The parameter '{}' is not belongs to any cell,"
                             "the data of parameter cannot be exported.".format(param_proto.name))
    incremental = kwargs.get('incremental', False)
    for map_param_proto in model.graph.map_parameter:
        map_param_name = map_param_proto.name[map_param_proto.name.find(":") + 1:]
        if map_param_name in net_dict.keys():
            map_parameter = net_dict[map_param_name]
            key_bytes, value_bytes, status_bytes = map_parameter.export_bytes(incremental)
            map_param_proto.key_tensor.raw_data = key_bytes
            map_param_proto.value_tensor.raw_data = value_bytes
            map_param_proto.status_tensor.raw_data = status_bytes
        else:
            raise ValueError("The map_parameter '{}' is not belongs to any cell,"
                             "the data of parameter cannot be exported.".format(map_param_proto.name))
    if not file_name.endswith('.mindir'):
        file_name += ".mindir"
    current_path = os.path.abspath(file_name)
    dirname = os.path.dirname(current_path)
    os.makedirs(dirname, exist_ok=True)
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWUSR)
    with open(file_name, 'wb') as f:
        os.chmod(file_name, stat.S_IRUSR | stat.S_IWUSR)
        model_string = model.SerializeToString()
        if is_encrypt():
            if callable(kwargs.get('enc_mode')):
                enc_func = kwargs.get('enc_mode')
                model_string = enc_func(model_string, kwargs.get('enc_key'))
            else:
                model_string = _encrypt(model_string, len(model_string), kwargs.get('enc_key'),
                                        len(kwargs.get('enc_key')), kwargs.get('enc_mode'))
        f.write(model_string)
        os.chmod(file_name, stat.S_IRUSR)


def _save_together(net_dict, model):
    """Whether graph and parameter save together during save mindir model."""
    data_total = 0
    for param_proto in model.graph.parameter:
        name = param_proto.name[param_proto.name.find(":") + 1:]
        if name in net_dict.keys():
            data_total += sys.getsizeof(net_dict[name].data.get_bytes()) / 1024
        else:
            raise ValueError("The parameter '{}' is not belongs to any cell,"
                             "the data of parameter cannot be exported.".format(param_proto.name))
        if data_total > TOTAL_SAVE:
            return False
    return True


def _save_dataset_to_mindir(model, dataset):
    """Save dataset preprocess operations into mindir model."""
    dataset_json = dataset.to_json()
    reverse_dataset = []
    while dataset_json:
        reverse_dataset = [dataset_json] + reverse_dataset
        if len(dataset_json['children']) > 1:
            logger.warning("Need to support dataset_node with more than one child, using child 0 as default.")
        dataset_json = dataset_json['children'][0] if dataset_json['children'] else []

    for op in reverse_dataset:
        if op['op_type'] == 'Map':
            model.preprocessor.op.add()
            model.preprocessor.op[-1].input_columns = json.dumps(op['input_columns'])
            model.preprocessor.op[-1].output_columns = json.dumps(op['output_columns'])
            model.preprocessor.op[-1].op_type = json.dumps(op['op_type'])
            model.preprocessor.op[-1].operations = json.dumps(op['operations'])
            model.preprocessor.op[-1].offload = op['offload'] if 'offload' in op.keys() else False


def parse_print(print_file_name):
    """
    Parse data file generated by :class:`mindspore.ops.Print`.

    Args:
        print_file_name (str): The file name needs to be parsed.

    Returns:
        List, element of list is Tensor.

    Raises:
        ValueError: The print file does not exist or is empty.
        RuntimeError: Failed to parse the file.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor, ops
        >>> ms.set_context(mode=ms.GRAPH_MODE, print_file_path='log.data')
        >>> class PrintInputTensor(nn.Cell):
        ...         def __init__(self):
        ...             super().__init__()
        ...             self.print = ops.Print()
        ...
        ...         def construct(self, input_pra):
        ...             self.print('print:', input_pra)
        ...             return input_pra
        >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
        >>> input_pra = Tensor(x)
        >>> net = PrintInputTensor()
        >>> net(input_pra)
        >>>
        >>> data = ms.parse_print('./log.data')
        >>> print(data)
        ['print:', Tensor(shape=[2, 4], dtype=Float32, value=
        [[ 1.00000000e+00,  2.00000000e+00,  3.00000000e+00,  4.00000000e+00],
        [ 5.00000000e+00,  6.00000000e+00,  7.00000000e+00,  8.00000000e+00]])]
    """
    print_file_path = os.path.abspath(print_file_name)

    if os.path.getsize(print_file_path) == 0:
        raise ValueError("For 'parse_print', the print file may be empty, please make sure enter the correct "
                         "'print_file_name'.")

    logger.info("Execute load print process.")
    print_list = Print()

    try:
        with open(print_file_path, "rb") as f:
            pb_content = f.read()
        print_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.critical("Failed to read the print file %s, please check whether the file is "
                        "correct.", print_file_name)
        raise ValueError(e.__str__() + "\nFailed to read the print file {}, please check whether "
                                       "the file is correct.".format(print_file_name)) from e

    tensor_list = []

    try:
        for print_ in print_list.value:
            # String type
            if print_.HasField("desc"):
                tensor_list.append(print_.desc)
            elif print_.HasField("tensor"):
                dims = print_.tensor.dims
                data_type = print_.tensor.tensor_type
                data = print_.tensor.tensor_content
                np_type = tensor_to_np_type.get(data_type)
                param_data = np.fromstring(data, np_type)
                ms_type = tensor_to_ms_type.get(data_type)
                if dims and dims != [0]:
                    param_value = param_data.reshape(dims)
                    tensor_list.append(Tensor(param_value, ms_type))
                # Scalar type
                else:
                    data_type_ = data_type.lower()
                    if 'float' in data_type_:
                        param_data = float(param_data[0])
                    elif 'int' in data_type_:
                        param_data = int(param_data[0])
                    elif 'bool' in data_type_:
                        param_data = bool(param_data[0])
                    tensor_list.append(Tensor(param_data, ms_type))

    except BaseException as e:
        logger.critical("Failed to load the print file %s.", print_list)
        raise RuntimeError(e.__str__() + "\nFailed to load the print file {}.".format(print_list)) from e

    return tensor_list


def _merge_param_with_strategy(sliced_data, parameter_name, strategy, is_even):
    """
    Merge data slices to one tensor with whole data when strategy is not None.

    Args:
        sliced_data (list[numpy.ndarray]): Data slices in order of rank_id.
        parameter_name (str): Name of parameter.
        strategy (dict): Parameter slice strategy.
        is_even (bool): Slice manner that True represents slicing evenly and False represents slicing unevenly.

    Returns:
        Tensor, the merged Tensor which has the whole data.

    Raises:
        ValueError: Failed to merge.
    """
    layout = strategy.get(parameter_name)
    try:
        dev_mat = list(layout.dev_matrix[0].dim)
        tensor_map = list(layout.tensor_map[0].dim)
        param_split_shape = list(layout.param_split_shape[0].dim)
        field_size = int(layout.field)
    except BaseException as e:
        raise ValueError(f"{e.__str__()}. For 'merge_sliced_parameter'"
                         f", please make sure that 'strategy' is correct.") from e

    device_count = 1
    for dim in dev_mat:
        device_count *= dim

    if len(sliced_data) != device_count:
        raise ValueError(f"For 'merge_sliced_parameter', the length of 'sliced_parameters' should be equal to "
                         f"device_count. The length of 'sliced_parameters' is {len(sliced_data)}, but "
                         f"device_count is {device_count}.")

    if not param_split_shape:
        if not is_even:
            raise ValueError("For 'merge_sliced_parameter', the shape of every parameter in 'sliced_parameters' "
                             "should be the same when slice manner is even.")

        all_gather_tensor = Tensor(np.concatenate(sliced_data))

        if field_size > 0:
            merged_tensor = _reshape_param_data_with_weight(all_gather_tensor, dev_mat, field_size)
        else:
            merged_tensor = _reshape_param_data(all_gather_tensor, dev_mat, tensor_map)

    else:
        tensor_strategy = _get_tensor_strategy(dev_mat, tensor_map)

        slice_count = 1
        for dim in tensor_strategy:
            slice_count *= dim

        if len(param_split_shape) != slice_count:
            raise ValueError(f"For 'merge_sliced_parameter', the param_split_shape length in 'strategy' should be "
                             f"{slice_count}, but got {len(param_split_shape)}.")

        tensor_slices_new = list(range(slice_count))
        tensor_slices = sliced_data
        for i in range(device_count):
            slice_index = int(_get_tensor_slice_index(dev_mat, tensor_strategy, tensor_map, i))
            if tensor_slices[i].shape[0] != param_split_shape[slice_index]:
                raise ValueError(f"For 'merge_sliced_parameter', the slice {slice_index} should be "
                                 f"{param_split_shape[slice_index]} in 0 axis, but got "
                                 f"{tensor_slices[i].shape[0]}.")
            tensor_slices_new[slice_index] = np.array(tensor_slices[i])

        dim_len = len(tensor_strategy)
        for i in range(dim_len):
            ele_count = int(len(tensor_slices_new) / tensor_strategy[dim_len - 1 - i])
            tensor_slices_new_inner = []
            for j in range(ele_count):
                new_tensor = tensor_slices_new[j * tensor_strategy[dim_len - 1 - i]]
                for k in range(j * tensor_strategy[dim_len - 1 - i] + 1,
                               (j + 1) * tensor_strategy[dim_len - 1 - i]):
                    new_tensor = np.concatenate((new_tensor, tensor_slices_new[k]), axis=dim_len - 1 - i)
                tensor_slices_new_inner.insert(len(tensor_slices_new_inner), np.array(new_tensor))
            tensor_slices_new = tensor_slices_new_inner
        merged_tensor = Tensor(tensor_slices_new[0])

    return merged_tensor


def restore_group_info_list(group_info_file_name):
    """
    Build rank list, the checkpoint of ranks in the rank list has the same contents with the local rank
    who saves the `group_info_file_name`. To save the group info file, please export GROUP_INFO_FIL
    environment variables like "export GROUP_INFO_FILE=/data/group_info.pb".

    Args:
        group_info_file_name (str): Name of group information file.

    Returns:
        List, the rank list.

    Raises:
        ValueError: group information file is incorrect.
        TypeError: `group_info_file_name` is not str.

    Examples:
        >>> import mindspore as ms
        >>> ms.restore_list = restore_group_info_list("./group_info.pb")
    """
    if not isinstance(group_info_file_name, str):
        raise TypeError(f"For 'restore_group_info_list', the argument 'group_info_file_name' should be str, "
                        f"but got {type(group_info_file_name)}.")

    if not os.path.isfile(group_info_file_name):
        raise ValueError(f"For 'restore_group_info_list', no such group information file: {group_info_file_name}.")

    if os.path.getsize(group_info_file_name) == 0:
        raise ValueError("For 'restore_group_info_list', the group information file should not be empty.")

    return _restore_group_info_list(group_info_file_name)


def build_searched_strategy(strategy_filename):
    """
    Build strategy of every parameter in network. Used in the case of distributed inference.

    Args:
        strategy_filename (str): Name of strategy file.

    Returns:
        Dict, whose key is parameter name and value is slice strategy of this parameter.

    Raises:
        ValueError: Strategy file is incorrect.
        TypeError: `strategy_filename` is not a string.

    Examples:
        >>> import mindspore as ms
        >>> strategy = ms.build_searched_strategy("./strategy_train.ckpt")
    """
    return _build_searched_strategy(strategy_filename)


def merge_sliced_parameter(sliced_parameters, strategy=None):
    """
    Merge parameter slices into one parameter. Used in the case of distributed inference.

    Args:
        sliced_parameters (list[Parameter]): Parameter slices in order of rank id.
        strategy (Optional[dict]): Parameter slice strategy, whose key is parameter name and
            value is slice strategy of this parameter. If strategy is None, just merge
            parameter slices in 0 axis order. Default: ``None``.

    Returns:
        Parameter, the merged parameter which has the whole data.

    Raises:
        ValueError: Failed to merge.
        TypeError: The sliced_parameters is incorrect or strategy is not dict.
        KeyError: The parameter name is not in keys of strategy.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, Parameter
        >>>
        >>> sliced_parameters = [
        ...                      Parameter(Tensor(np.array([0.00023915, 0.00013939, -0.00098059])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00015815, 0.00015458, -0.00012125])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00042165, 0.00029692, -0.00007941])),
        ...                                "network.embedding_table"),
        ...                      Parameter(Tensor(np.array([0.00084451, 0.00089960, -0.00010431])),
        ...                                "network.embedding_table")]
        >>> merged_parameter = ms.merge_sliced_parameter(sliced_parameters)
        >>> print(merged_parameter)
        Parameter (name=network.embedding_table, shape=(12,), dtype=Float64, requires_grad=True)
    """
    if not isinstance(sliced_parameters, list):
        raise TypeError(f"For 'merge_sliced_parameter', the argument 'sliced_parameters' should be list, "
                        f"but got {type(sliced_parameters)}.")

    if not sliced_parameters:
        raise ValueError("For 'merge_sliced_parameter', the argument 'sliced_parameters' should not be empty.")

    if strategy and not isinstance(strategy, dict):
        raise TypeError(f"For 'merge_sliced_parameter', the argument 'strategy' should be dict, "
                        f"but got {type(strategy)}.")

    try:
        parameter_name = sliced_parameters[0].name
        parameter_shape = sliced_parameters[0].data.shape
        parameter_shape_length = len(parameter_shape)
    except BaseException as e:
        raise TypeError(e.__str__() + f" For 'merge_sliced_parameter', the element in 'sliced_parameters' should be "
                                      f"'Parameter', but got {type(sliced_parameters[0])} at index 0.") from e

    is_even = True
    for index, parameter in enumerate(sliced_parameters):
        if not isinstance(parameter, Parameter):
            raise TypeError(f"For 'merge_sliced_parameter', the element in 'sliced_parameters' should be 'Parameter', "
                            f"but got {type(parameter)} at index {index}.")

        if parameter.name != parameter_name \
                or len(parameter.data.shape) != parameter_shape_length \
                or parameter.data.shape[1:] != parameter_shape[1:]:
            raise ValueError(f"For 'merge_sliced_parameter', please make sure that the elements in 'slice_parameters'"
                             f" have the same name, dimension length and shape except 0 axis. The name, dimension "
                             f"length, shape except 0 axis should be {parameter_name}, {parameter_shape_length}, "
                             f"{parameter_shape[1:]}, but got name: {parameter.name}, dimension length: "
                             f"{len(parameter.data.shape)}, shape except 0 axis: {parameter.data.shape[1:]} "
                             f"at index {index}.")

        if parameter.data.shape != parameter_shape:
            is_even = False

    layerwise_parallel = sliced_parameters[0].layerwise_parallel
    requires_grad = sliced_parameters[0].requires_grad
    sliced_data = [parameter.data.asnumpy() for parameter in sliced_parameters]

    if not strategy:
        merged_tensor = Tensor(np.concatenate(sliced_data))
        merged_parameter = Parameter(merged_tensor, parameter_name, requires_grad, layerwise_parallel)

    else:
        if parameter_name not in strategy.keys():
            raise KeyError(f"For 'merge_sliced_parameter', the parameter name {parameter_name} should be a key in "
                           f"the 'strategy'. Please check 'sliced_parameter' and 'strategy'.")
        merged_tensor = _merge_param_with_strategy(sliced_data, parameter_name, strategy, is_even)
        merged_parameter = Parameter(merged_tensor, parameter_name, requires_grad, layerwise_parallel)

    return merged_parameter


def load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None,
                                train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM'):
    """
    Load checkpoint into net for distributed predication. Used in the case of distributed inference.

    Args:
        network (Cell): Network for distributed predication.
        checkpoint_filenames (list[str]): The name of Checkpoint files in order of rank id.
        predict_strategy (dict): Strategy of predication process. It means that using one device to predict
                                 when setting predict_strategy as None. Default: ``None`` .
        train_strategy_filename (str): The filename of training strategy protocol buffer file.
                                       When train_strategy_filename is None, the training strategy file will be
                                       obtained from context.get_auto_parallel_context("strategy_ckpt_load_file").
                                       Therefore, the training strategy file needs to be specified
                                       in at least one of them. Default: ``None`` .
        strict_load (bool): Whether to strict load the parameter into net. If ``False`` , it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: ``False`` .
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is ``None`` , the decryption
                                      is not required. Default: ``None`` .
        dec_mode (str): This parameter is valid only when dec_key is not set to ``None`` . Specifies the decryption
                        mode, currently supports ``'AES-GCM'`` , ``'AES-CBC'``  and ``'SM4-CBC'`` .
                        Default: ``'AES-GCM'`` .

    Raises:
        TypeError: The type of inputs do not match the requirements.
        ValueError: Failed to load checkpoint into net.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For the Ascend devices, users need to prepare the rank table, set rank_id and device_id.
            Please see the `rank table startup
            <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/rank_table.html>`_
            for more details.

            For the GPU devices, users need to prepare the host file and mpi, please see the `mpirun startup
            <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/mpirun.html>`_ .

            For the CPU device, users need to write a dynamic cluster startup script, please see the `Dynamic Cluster
            Startup <https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/dynamic_cluster.html>`_ .

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, train
        >>> from mindspore.communication import init
        >>>
        >>> step_per_epoch = 4
        >>> device_num = 8
        >>>
        >>> # Define the network structure.
        >>> class Net(nn.Cell):
        ...     def __init__(self, matmul_size, strategy=None):
        ...         super().__init__()
        ...         matmul_np = np.full(matmul_size, 0.5, dtype=np.float32)
        ...         self.matmul_weight = ms.Parameter(ms.Tensor(matmul_np))
        ...         self.matmul = ops.MatMul()
        ...         self.neg = ops.Neg()
        ...         if strategy is not None:
        ...             self.matmul.shard(strategy)
        ...
        ...     def construct(self, inputs):
        ...         x = self.matmul(inputs, self.matmul_weight)
        ...         x = self.neg(x)
        ...         return x
        >>>
        >>> # Create dataset.
        >>> def get_dataset(*inputs):
        ...     def generate():
        ...         for _ in range(step_per_epoch):
        ...             yield inputs
        ...     return generate
        >>>
        >>> # Train network and save distributed checkpoint.
        >>> def train_net():
        ...     ms.set_context(mode=ms.GRAPH_MODE)
        ...     init()
        ...     np.random.seed(1)
        ...     input_data = np.random.rand(16, 96).astype(np.float32)
        ...     label_data = np.random.rand(16, 16).astype(np.float32)
        ...     fake_dataset = get_dataset(input_data, label_data)
        ...     dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
        ...
        ...     # Set parallel strategy.
        ...     strategy = ((1, 4), (4, 1))
        ...     ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_num,
        ...                                  strategy_ckpt_save_file="./train_strategy.ckpt")
        ...     network = Net(matmul_size=(96, 16), strategy=strategy)
        ...     net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        ...     net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
        ...     model = ms.Model(network=network, loss_fn=net_loss, optimizer=net_opt)
        ...     ckpt_config = train.CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
        ...     global_rank_id = int(os.getenv("RANK_ID"))
        ...     ckpt_path = "./rank_{}_ckpt".format(global_rank_id)
        ...     ckpt_callback = train.ModelCheckpoint(prefix="parallel", directory=ckpt_path, config=ckpt_config)
        ...     model.train(epoch=2, train_dataset=dataset, callbacks=[ckpt_callback], dataset_sink_mode=False)
        ...     ms.reset_auto_parallel_context()
        >>>
        >>> # Load distributed checkpoint and test.
        >>> def load_model():
        ...     ms.set_context(mode=ms.GRAPH_MODE)
        ...     init()
        ...     ms.set_auto_parallel_context(full_batch=True, parallel_mode="semi_auto_parallel",
        ...                                  strategy_ckpt_load_file="./train_strategy.ckpt", device_num=device_num)
        ...     predict_data = ms.Tensor(np.random.randn(128, 96).astype(np.float32))
        ...     network = Net(matmul_size=(96, 16))
        ...     model = ms.Model(network)
        ...     predict_layout = model.infer_predict_layout(ms.Tensor(predict_data))
        ...     ckpt_file_list = ["./rank_{}_ckpt/parallel-2_4.ckpt".format(i) for i in range(0, device_num)]
        ...     ms.load_distributed_checkpoint(network, ckpt_file_list, predict_layout)
        ...     predict_result = model.predict(predict_data)
        ...     print(predict_result)
        >>>
        >>> train_net()
        >>> load_model()
        [[-7.3259363 -7.497216  -7.398196  ... -7.374962  -7.204874  -7.234935 ]
        [ 3.362938   3.3535435  3.3832688 ...  3.4263954  3.279045   3.3202887]
        ...
        [ 1.6067538  1.6244187  1.5384722 ...  1.5449994  1.6195512  1.6176052]]
    """
    network = Validator.check_isinstance("network", network, nn.Cell)
    _check_checkpoint_file(checkpoint_filenames)
    _check_predict_strategy(predict_strategy)

    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)

    if train_strategy_filename is None:
        train_strategy_filename = context.get_auto_parallel_context("strategy_ckpt_load_file")
    _train_strategy = build_searched_strategy(train_strategy_filename)
    train_strategy = _convert_to_list(_train_strategy)

    train_dev_count = 1
    ckpt_file_len = len(checkpoint_filenames)
    for dim in train_strategy[list(train_strategy.keys())[0]][0]:
        train_dev_count *= dim
    if train_dev_count != ckpt_file_len:
        raise ValueError(f"For 'Load_distributed_checkpoint', the length of 'checkpoint_filenames' should be "
                         f"equal to the device count of training process. "
                         f"But got the length of 'checkpoint_filenames'"
                         f" is {ckpt_file_len} and the device count is {train_dev_count}.")
    rank_list = _infer_rank_list(train_strategy, predict_strategy)

    param_total_dict = defaultdict(dict)
    for file_index, file_name in enumerate(checkpoint_filenames):
        ckpt_dict = load_checkpoint(file_name, dec_key=dec_key, dec_mode=dec_mode)
        for param_name, param in ckpt_dict.items():
            param_total_dict[param_name][file_index] = param

    param_dict = {}
    param_not_in_strategy = []
    param_not_in_ckpt = []
    for _, param in network.parameters_and_names():
        sliced_params = []
        if param.name not in rank_list.keys():
            param_not_in_strategy.append(param.name)
            continue
        if param.name not in param_total_dict:
            param_not_in_ckpt.append(param.name)
            continue

        param_rank = rank_list.get(param.name)[0]
        skip_merge_split = rank_list.get(param.name)[1]
        shard_stride = train_strategy.get(param.name)[4]
        if train_strategy.get(param.name)[5]:
            shard_size = ckpt_file_len / shard_stride / train_strategy.get(param.name)[5]
        else:
            shard_size = 0
        for rank in param_rank:
            param_total_list = list(range(0, ckpt_file_len))
            if shard_size > 0:
                shard_total_list = []
                for i in range(0, ckpt_file_len, shard_size):
                    shard_total_list.append(param_total_list[i:i + shard_size])
                param_total_list = shard_total_list[rank // shard_size]
            if shard_stride > 0:
                param_stride = []
                # merge pre parameter
                param_index = param_total_list[0:param_total_list.index(rank) + 1][::-1][::shard_stride]
                param_index.extend(param_total_list[param_total_list.index(rank):][::shard_stride])
                param_index = list(set(param_index))
                param_index.sort()
                for rank_num in param_index:
                    param_stride.append(param_total_dict[param.name][rank_num].data.asnumpy())

                sliced_param = Parameter(Tensor(np.concatenate(param_stride)), name=param.name)
            else:
                sliced_param = param_total_dict[param.name][rank]

            sliced_params.append(sliced_param)
        if skip_merge_split:
            split_param = sliced_params[0]
        else:
            param_unique_strategy = _remove_repeated_slices(train_strategy[param.name])
            _param_unique_strategy = _convert_to_layout(param.name, param_unique_strategy)
            split_param = _merge_and_split(sliced_params, _param_unique_strategy, predict_strategy)
        opt_shard_group = predict_strategy[param.name][5] if predict_strategy else None
        if opt_shard_group:
            data = split_param.data.asnumpy()
            rank = get_rank(opt_shard_group)
            size = get_group_size(opt_shard_group)
            try:
                data_slice = np.split(data, size)[rank]
            except BaseException as e:
                logger.critical("Failed to load opt shard slice in load distributed checkpoint for {}. Data shape is {}"
                                " and group is {}".format(param.name, split_param.data.shape, opt_shard_group))
                raise RuntimeError(e.__str__() + f"\nFor 'load_distributed_checkpoint', failed to load opt shard slice"
                                                 f" in load distributed checkpoint for {param.name}. Data shape is "
                                                 f"{split_param.data.shape} and group is {opt_shard_group}.") from e
            split_param = Parameter(Tensor(data_slice), param.name,
                                    split_param.requires_grad, split_param.layerwise_parallel)
        param_dict[param.name] = split_param

    if param_not_in_strategy:
        logger.warning("For 'load_distributed_checkpoint', {} parameters in network are not in the slice strategy, "
                       "you can check whether 'predict_strategy' or 'train_strategy_filename' is correct."
                       .format(param_not_in_strategy))
    if param_not_in_ckpt:
        logger.warning("For 'load_distributed_checkpoint', {} parameters in network and slice strategy but not in "
                       "the checkpoint file, please check whether 'checkpoint_filenames' is correct."
                       .format(param_not_in_ckpt))

    load_param_into_net(network, param_dict, strict_load=strict_load)


def async_ckpt_thread_status():
    """
    Get the status of asynchronous save checkpoint thread.

    When performing asynchronous save checkpoint, you can determine whether the asynchronous thread is completed.

    Returns:
        bool, True, Asynchronous save checkpoint thread is running.
        False, Asynchronous save checkpoint thread is not executing.

    Examples:
        >>> import mindspore as ms
        >>> ms.async_ckpt_thread_status()
        False
    """
    thr_list = threading.enumerate()
    return True in [ele.getName() == "asyn_save_ckpt" for ele in thr_list]


def _check_predict_strategy(predict_strategy):
    """Check predict strategy."""

    def _check_int_list(arg):
        if not isinstance(arg, list):
            return False
        for item in arg:
            if not isinstance(item, int):
                return False
        return True

    if predict_strategy is None:
        return

    flag = True
    predict_strategy = Validator.check_isinstance("predict_strategy", predict_strategy, dict)
    for key in predict_strategy.keys():
        if not isinstance(key, str) or not isinstance(predict_strategy[key], (list, tuple)) \
                or len(predict_strategy[key]) < 4:
            flag = False
        dev_matrix, tensor_map, param_split_shape, field_size = predict_strategy[key][:4]
        if not _check_int_list(dev_matrix) or not _check_int_list(tensor_map) or \
                not (_check_int_list(param_split_shape) or not param_split_shape) or \
                not (isinstance(field_size, int) and field_size == 0):
            flag = False

    if not flag:
        raise ValueError(f"For 'load_distributed_checkpoint', the argument 'predict_strategy' is dict, "
                         f"the key of it must be string, and the value of it must be list or tuple that "
                         f"the first four elements must be dev_matrix (list[int]), tensor_map (list[int]), "
                         f"param_split_shape (list[int]) and field_size (int, which value is 0)."
                         f"Please check whether 'predict_strategy' is correct.")


def _check_checkpoint_file(checkpoint_filenames):
    """Check checkpoint file name."""
    for index, filename in enumerate(checkpoint_filenames):
        if not isinstance(filename, str) or not os.path.exists(filename) \
                or filename[-5:] != ".ckpt" or os.path.getsize(filename) == 0:
            raise ValueError(f"For 'load_distributed_checkpoint', please check 'checkpoint_filenames', and "
                             f"make sure the {filename} at index {index} is a valid checkpoint file, it must "
                             f"be a string ending with '.ckpt', and the checkpoint file it represents must "
                             f"be exist and not empty.")


def _merge_and_split(sliced_params, train_strategy, predict_strategy):
    """Merge sliced parameter and split it according to the predict strategy."""
    merged_param = merge_sliced_parameter(sliced_params, train_strategy)
    if predict_strategy is None:
        return merged_param
    param_name = merged_param.name
    tensor_layout = predict_strategy[param_name]
    rank = get_rank()
    split_tensor = _load_tensor(merged_param.data, tensor_layout[0], tensor_layout[1], rank)
    requires_grad = merged_param.requires_grad
    layerwise_parallel = merged_param.layerwise_parallel
    split_param = Parameter(split_tensor, param_name, requires_grad, layerwise_parallel)
    return split_param


def _calculation_net_size(net):
    """Calculate the size of parameters in the network."""
    data_total = 0
    net_dict = net.parameters_dict()
    for name in net_dict:
        data_total += sys.getsizeof(net_dict[name].data.asnumpy().tobytes()) / 1024

    return data_total


def _get_mindir_inputs(file_name):
    """
    Get MindIR file's inputs.

    Note:
        1. Parsing encrypted MindIR file is not supported.
        2. Parsing dynamic shape MindIR file is not supported.

    Args:
        file_name (str): MindIR file name.

    Returns:
        Tensor, list(Tensor), the input of MindIR file.

    Raises:
        TypeError: If the parameter file_name is not `str`.
        RuntimeError: MindIR's input is not tensor type or has no dims.

    Examples:
        >>> input_tensor = get_mindir_inputs("lenet.mindir")
    """
    Validator.check_file_name_by_regular(file_name)
    file_name = os.path.abspath(file_name)
    model = read_proto(file_name)
    input_tensor = []

    for ele_input in model.graph.input:
        input_shape = []
        if not hasattr(ele_input, "tensor") or not hasattr(ele_input.tensor[0], "dims"):
            raise RuntimeError("MindIR's inputs has no tensor or tensor has no dims, please check MindIR file.")

        for ele_shape in ele_input.tensor[0].dims:
            input_shape.append(ele_shape)
        if is_shape_unknown(input_shape):
            raise RuntimeError(f"MindIR input's shape is: {input_shape}, dynamic shape is not supported.")

        mindir_type = ele_input.tensor[0].data_type
        if mindir_type not in mindir_to_tensor_type:
            raise RuntimeError(f"MindIR input's type: {mindir_type} is not supported.")

        input_type = mindir_to_tensor_type.get(mindir_type)
        input_tensor.append(Tensor(shape=input_shape, dtype=input_type, init=One()))

    if not input_tensor:
        logger.warning("The MindIR model has no input, return None.")
        return None
    return input_tensor[0] if len(input_tensor) == 1 else input_tensor


def convert_model(mindir_file, convert_file, file_format):
    """
    Convert mindir model to other format model. Current version only support convert to "ONNX" format.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        mindir_file (str): MindIR file name.
        convert_file (str): Convert model file name.
        file_format (str): Convert model's format, current version only supports "ONNX".

    Raises:
        TypeError: If the parameter `mindir_file` is not `str`.
        TypeError: If the parameter `convert_file` is not `str`.
        ValueError: If the parameter `file_format` is not "ONNX".

    Examples:
        >>> import mindspore as ms
        >>> ms.convert_model("lenet.mindir", "lenet.onnx", "ONNX")
    """
    Validator.check_file_name_by_regular(mindir_file)
    Validator.check_file_name_by_regular(convert_file)
    if file_format != "ONNX":
        raise ValueError(f"For 'convert_model', 'file_format' must be 'ONNX', but got {file_format}.")
    net_input = _get_mindir_inputs(mindir_file)
    graph = load(mindir_file)
    net = nn.GraphCell(graph)
    if isinstance(net_input, Tensor):
        export(net, net_input, file_name=convert_file, file_format=file_format)
    else:
        export(net, *net_input, file_name=convert_file, file_format=file_format)
