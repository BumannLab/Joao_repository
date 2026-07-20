import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from Scripts.NeuralNetworkSegmentation.optimizer import create_optimizer, create_lr_scheduler
from Scripts.NeuralNetworkSegmentation.models.unet import get_model
from Scripts.NeuralNetworkSegmentation.loss import get_loss
from Scripts.NeuralNetworkSegmentation.config import load_config_yaml, save_config_yaml
from Scripts.NeuralNetworkSegmentation.dataset.data import get_train_loaders
from Scripts.NeuralNetworkSegmentation.train import ModelTrainer
from Scripts.NeuralNetworkSegmentation.metrics import get_eval_criterion


CONFIG_PATH = "Results/MycobacteriaTuberculosisSegmentation/TestNewScript/previous2dpipeline.yml"

def main(
        config_path,
):  
    # TODO: Build tests for all of this
    config = load_config_yaml(config_path)
    
    # Create the model
    model = get_model(config["model"])

    # Read device and move model to device
    device = config.get("device", None)
    assert device, "Device should be specified in the config file"
    model.to(device)

    # Create the optimizer
    optimizer = create_optimizer(config["optimizer"], model)
    
    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get("lr_scheduler", None), optimizer) 
    
    # Create loss criterion
    loss_criterion = get_loss(config["loss_criterion"])

    # Create evaluation metric
    eval_criterion = get_eval_criterion(config["eval_criterion"])

    # Create data loaders
    loaders = get_train_loaders(config["loaders"])

    train_config = config["trainer"]
    resume = train_config.pop("resume", None)
    pre_trained = train_config.pop("pre_trained", None)
    # Create trainer

    trainer = ModelTrainer(
        model = model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        loaders=loaders,
        resume=resume,
        pre_trained=pre_trained,
        **train_config,
        device = device,
    )

    trainer.fit()


if __name__ == "__main__":
    main(CONFIG_PATH)
