import yaml
import numpy as np
import torch
from torch import nn
from importlib import import_module

from greycat import *


def get_config(config_path: str) -> dict:
    """
    Reads the config.yaml as a dict.
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_model(config: dict, device: torch.device) -> nn.Module:
    """
    Gets the selected torch model from the corresponding local Python module and using the config parameters.
    """
    lib = config["model_selection"]["model_lib"]
    name = config["model_selection"]["model_name"]

    module = import_module("lib." + lib)
    class_obj = getattr(module, name)

    parameters = config[name]
    parameters["n_features"] = config["n_features"]
    model = class_obj(parameters)
    model = model.float().to(device)

    return model


def push_results(config_path: str, model_path: str, history: dict, greycat: GreyCat):

    train_history = np.array(history["train"])
    test_history = np.array(history["val"])

    history_numpy = np.column_stack([train_history, test_history])

    history_table = std.core.Table.from_numpy(greycat, history_numpy)
    
    greycat.call("project::saveModel", [config_path, model_path, history_table])