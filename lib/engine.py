import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from termcolor import colored


from greycat import *
from lib import processing


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
    