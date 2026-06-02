import torch

from ZOrganized_scripts.NeuralNetworkSegmentationConfig.optimizer import get_optimizer, get_lr_scheduler
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.models.unet import get_model
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.loss import get_loss
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.config import load_config_yaml
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.data import build_dataloaders
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.train import ModelTrainer
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.utils import build_seed_from_time, get_save_dir_model_name
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.metrics import get_eval_criterion


CONFIG_PATH = "/data/joao/Materials/Joao_repository/ZOrganized_scripts/NeuralNetworkSegmentationConfig/config.yml"


def main(
        config_path,
):  
    # TODO: Build tests for all of this
    config = load_config_yaml(config_path)

    save_dir, model_name = get_save_dir_model_name(config)

    seed = config.pop("seed", build_seed_from_time()) # If seed does not exist create one and append it to config
    config["seed"] = seed
    
    device = config.pop("device", torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu") # If device does search whether gpu exists and use it if not use cpu
    config["device"] = device # write it to config

    model = get_model(config["model"]).to(device)

    optimizer = get_optimizer(config["optimizer"], model)
    lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer)
 
    loss_criterion = get_loss(config["loss_criterion"])
    eval_criterion = get_eval_criterion(config["eval_criterion"])

    loaders = build_dataloaders(config["data"], config["transforms"])

    num_epochs = config["num_epochs"] 

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
