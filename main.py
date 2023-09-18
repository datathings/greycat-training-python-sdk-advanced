import os
import json
import yaml
import random
import numpy as np
import torch
from torch import nn
from importlib import import_module
from datetime import datetime
from tqdm import tqdm
from termcolor import colored


from greycat import *
from lib import processing


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


def train_loop(model: nn.Module, train_dataset: processing.TrainDataset, val_size: float, n_epochs: int, optimizer: torch.optim, criterion: torch.nn.modules.loss, device: torch.device, shuffle: bool = True) -> tuple[nn.Module, dict]:
    """
    Given the train loop parameters, perform several epochs of training and optimizes the weight. For each epoch there is a validation round.
    """
    last_train_index = int((1-val_size)*train_dataset.max_index)

    history = dict(train = [], val = [])

    for epoch in range(1, n_epochs+1):
        model.train()
        train_losses = []
        batches = list(np.linspace(0, last_train_index, last_train_index+1))

        if shuffle:
            random.shuffle(batches)

        progress = tqdm(batches, colour="#00ff00", desc=colored(f"Epoch {epoch} train", "green"))
        for index in progress:
            item = train_dataset.get(int(index))
            x_in = item["x"]
            x_gt = item["y"].to(device)
            x_in = x_in.float()
            optimizer.zero_grad()
            x_in = x_in.to(device)
            x_out = model.forward(x_in)

            loss = criterion(x_gt, x_out)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
            progress.set_postfix_str(colored(f"Loss: {round(np.mean(train_losses), 3)}", "green"))

        train_loss = np.mean(train_losses)
        history["train"].append(train_loss)
        model.eval()
        val_losses = []

        with torch.no_grad():
            batches = list(np.linspace(last_train_index, train_dataset.max_index-1, train_dataset.max_index-last_train_index))

            if shuffle:
                random.shuffle(batches)

            progress = tqdm(batches, colour="#0000ff", desc=colored(f"Epoch {epoch} valid", "blue"))
            for index in progress:
                item = train_dataset.get(int(index))
                x_in = item["x"]
                x_gt = item["y"].to(device)
                x_in = x_in.float()
                x_in = x_in.to(device)
                x_out = model.forward(x_in)

                loss = criterion(x_gt, x_out)
                val_losses.append(loss.item())

                progress.set_postfix_str(colored(f"Loss: {round(np.mean(val_losses), 3)}", "blue"))

        val_loss = np.mean(val_losses)
        history["val"].append(val_loss)
        del progress

    return model, history


def train_model(model: nn.Module, train_data: processing.TrainDataset, config: dict, device: torch.device):
    """
    Trains the model with generated dataset (train_data) and using the config parameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = config["training"]["learning_rate"])
    criterion = torch.nn.L1Loss(reduction="mean").to(device)

    trained_model, history = train_loop(
        model=model,
        train_dataset=train_data,
        val_size=config["training"]["val_size"],
        n_epochs=config["training"]["epochs"],
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    return trained_model, history


def push_results(config_path: str, model_path: str, history: dict, greycat: GreyCat):

    train_history = np.array(history["train"])
    test_history = np.array(history["val"])

    history_numpy = np.column_stack([train_history, test_history])

    history_table = std.core.Table.from_numpy(greycat, history_numpy)
    
    greycat.call("project::saveModel", [config_path, model_path, history_table])



if __name__ == "__main__":
    config = get_config("config.yaml")
    
    if config["training"]["device"] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    greycat_session: GreyCat = GreyCat("http://localhost:8080")

    train_data = processing.TrainDataset(
        id = config["dataset_id"],
        greycat = greycat_session,
        n_rows=config["dataloader"]["n_rows"],
        batch_size=config["dataloader"]["batch_size"],
        window_len=config["dataloader"]["window_len"],
        delay=config["dataloader"]["delay"]
    )

    n_features = train_data.tensor_data.shape[1]
    config["n_features"] = n_features

    print(f"Batch shape: {train_data.get(0)['x'].shape}")
    model = get_model(config, device)

    trained_model, history = train_model(model, train_data, config, device)
    print("Training concluded successfully.")

    if config["training"]["save_model"]:
        now = datetime.today()
        model_name = config["model_selection"]["model_name"]
        model_path = f"trained-models/{model_name}__{now.year}-{now.month}-{now.day}--{now.hour}:{now.minute}:{now.second}.pth"
        history_path = f"histories/{model_name}__{now.year}-{now.month}-{now.day}--{now.hour}:{now.minute}:{now.second}.json"
        yaml_path = f"configs/{model_name}__{now.year}-{now.month}-{now.day}--{now.hour}:{now.minute}:{now.second}.yaml"

        torch.save(model, model_path)

        with open(history_path, "w") as outfile:
            json.dump(history, outfile, indent=4)
        
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        push_results(config_path=yaml_path, model_path=model_path, history=history, greycat=greycat_session)

        print(f"Trained model saved in {model_path}")