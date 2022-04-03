""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging

import torch
import torch.nn as nn

# from .features import FeatureListNet, FeatureDictNet, FeatureHookNet
# from .hub import has_hf_hub, download_cached_file, load_state_dict_from_hf, load_state_dict_from_url
# from .layers import Conv2dSame, Linear

_logger = logging.getLogger(__name__)

def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)



def fuse_bn_to_conv(bn_layer, conv_layer):
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = conv_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    conv_layer.weight.data.copy_(W)
    if conv_layer.bias is None:
        conv_layer.bias = torch.nn.Parameter(bias)
    else:
        conv_layer.bias.data.copy_(bias)


def fuse_bn_to_linear(bn_layer, linear_layer):
    # print('bn fuse')
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = linear_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    linear_layer.weight.data.copy_(W)
    if linear_layer.bias is None:
        linear_layer.bias = torch.nn.Parameter(bias)
    else:
        linear_layer.bias.data.copy_(bias)


def extract_layers(model):
    list_layers = []
    for n, p in model.named_modules():
        list_layers.append(n)
    return list_layers


def compute_next_bn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d':
        return next_bn
    return None


def compute_next_abn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'ABN':
        return next_bn
    return None


def compute_next_bn_1d(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm1d':
        return next_bn
    return None


def fuse_bn2d_bn1d_abn(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            next_bn = compute_next_bn(n, model)
            if next_bn is not None:
                next_bn_ = extract_layer(model, next_bn)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_bn, nn.Identity())

            next_abn = compute_next_abn(n, model)
            if next_abn is not None:
                next_bn_ = extract_layer(model, next_abn)
                activation = calc_abn_activation(next_bn_)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_abn, activation)
        if isinstance(m, torch.nn.Linear):
            next_bn1d = compute_next_bn_1d(n, model)
            if next_bn1d is not None:
                next_bn1d_ = extract_layer(model, next_bn1d)
                fuse_bn_to_linear(next_bn1d_, m)
                set_layer(model, next_bn1d, nn.Identity())

    return model

def calc_abn_activation(ABN_layer):
    from inplace_abn import ABN
    activation = nn.Identity()
    if isinstance(ABN_layer, ABN):
        if ABN_layer.activation == "relu":
            activation = nn.ReLU(inplace=True)
        elif ABN_layer.activation == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=ABN_layer.activation_param, inplace=True)
        elif ABN_layer.activation == "elu":
            activation = nn.ELU(alpha=ABN_layer.activation_param, inplace=True)
    return activation

def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    from models.layers import InplaceAbn
    from inplace_abn import ABN
    # convert all InplaceABN layer to bit-accurate ABN layers.
    if isinstance(module, InplaceAbn):
        module_new = ABN(module.num_features, activation=module.act_name,
                         activation_param=module.act_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module