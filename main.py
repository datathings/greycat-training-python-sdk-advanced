
import os
import json
import yaml
import torch
from datetime import datetime


from greycat import *
from lib import processing, engine, utils


if __name__ == "__main__":
    config = utils.get_config("config.yaml")
    
    if config["training"]["device"] == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    greycat_session: GreyCat = GreyCat("http://localhost:8080")

    train_data = processing.TrainDataset(
        id = config["dataset_id"],
        greycat = greycat_session,
        n_features=config["dataloader"]["n_features"],
        n_rows=config["dataloader"]["n_rows"],
        batch_size=config["dataloader"]["batch_size"],
        window_len=config["dataloader"]["window_len"],
        delay=config["dataloader"]["delay"]
    )

    n_features = train_data.tensor_data.shape[1]
    config["n_features"] = n_features

    print(f"Batch shape: {train_data.get(0)['x'].shape}")
    model = utils.get_model(config, device)

    trained_model, history = engine.train_model(model, train_data, config, device)
    print("Training concluded successfully.")

    if config["training"]["save_model"]:
        now = datetime.today()
        model_name = config["model_selection"]["model_name"]
        model_path = f"trained-models/{model_name}__{now.year}-{now.month}-{now.day}--{now.hour}:{now.minute}:{now.second}.pth"
        yaml_path = f"configs/{model_name}__{now.year}-{now.month}-{now.day}--{now.hour}:{now.minute}:{now.second}.yaml"

        for folder in ["trained-models", "configs"]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        torch.save(model, model_path)
        
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        utils.push_results(config_path=yaml_path, model_path=model_path, history=history, greycat=greycat_session)

        print(f"Trained model saved in {model_path}")