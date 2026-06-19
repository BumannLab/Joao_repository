import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch

from Scripts.NeuralNetworkSegmentation.optimizer import get_optimizer, get_lr_scheduler
from Scripts.NeuralNetworkSegmentation.models.unet import get_model
from Scripts.NeuralNetworkSegmentation.loss import get_loss
from Scripts.NeuralNetworkSegmentation.config import load_config_yaml, save_config_yaml
from Scripts.NeuralNetworkSegmentation.data import build_dataloaders
from Scripts.NeuralNetworkSegmentation.train import ModelTrainer
from Scripts.NeuralNetworkSegmentation.metrics import get_eval_criterion
from Scripts.NeuralNetworkSegmentation.utils import build_seed_from_time, get_save_dir_model_name, seed_everything


CONFIG_PATH = "/data/joao/Materials/Joao_repository/Config_mTB_segmentation/config_test_run.yml"

def main(
        config_path,
):  
    # TODO: Build tests for all of this
    config = load_config_yaml(config_path)

    save_dir, model_name = get_save_dir_model_name(config)

    seed = config.pop("seed", build_seed_from_time()) # If seed does not exist create one and append it to config
    config["seed"] = seed
    seed_everything(seed)
    
    device = config.pop("device", torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu") # If device does search whether gpu exists and use it if not use cpu
    config["device"] = device # write it to config

    save_config_yaml(save_dir, config)

    model = get_model(config["model"]).to(device)

    optimizer = get_optimizer(config["optimizer"], model)
    
    if "lr_scheduler" in config:
        lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer) 
    else:
        lr_scheduler = None

    loss_criterion = get_loss(config["loss_criterion"])
    eval_criterion = get_eval_criterion(config["eval_criterion"])

    loaders = build_dataloaders(config)

    num_epochs = config["trainer"]["num_epochs"] 

    trainer = ModelTrainer(
        save_dir = save_dir,
        save_model_name = model_name,
        model = model,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        loss_criterion = loss_criterion,
        eval_criterion = eval_criterion,
        loaders = loaders,
        max_num_epochs = num_epochs,
        device = device,
    )

    trainer.fit()



if __name__ == "__main__":
    main(CONFIG_PATH)
