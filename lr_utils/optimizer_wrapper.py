import torch
from torch import nn


def group_params(model, group_mode, initial_lr):
    if group_mode == "yolov5":
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # conv bias and bn bias
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # bn weight
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # conv weight
        params = [
            {"params": bias, "weight_decay": 0.0},
            {"params": norm, "weight_decay": 0.0},
            {"params": weight},
        ]
    elif group_mode == "all":
        params = model.parameters()
    elif group_mode == "finetune":
        if hasattr(model, "module"):
            model = model.module
        assert hasattr(model, "get_grouped_params"), "Cannot get the method get_grouped_params of the model."
        params_groups = model.get_grouped_params()
        params = [
            {"params": params_groups["pretrained"], "lr": 0.1 * initial_lr},
            {"params": params_groups["retrained"], "lr": initial_lr},
        ]
    else:
        raise NotImplementedError
    return params


def get_optimizer(model, args):
    params_group = group_params(model, group_mode=args.group_mode, initial_lr=args.learning_rate)
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params=params_group,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=params_group,
            lr=args.learning_rate,
            betas=(args.beta0, args.beta1),
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    else:
        raise NotImplementedError
    return optimizer
