from NeuralNetworkSegmentation.optimizer import get_optimizer, get_lr_scheduler
from NeuralNetworkSegmentation.models.unet import get_model
from NeuralNetworkSegmentation.loss import get_loss
from NeuralNetworkSegmentation.config import load_config_yaml
from NeuralNetworkSegmentation._data import BacteriaMaskPair

CONFIG_PATH = "/data/joao/Materials/Joao_repository/NeuralNetworkSegmentation/config.yml"


def main(
        config_path,
):
    
    config = load_config_yaml(config_path)

    data_set = load_dataset(config["data_path"])

    model = get_model(config["model"])
    optimizer = get_optimizer(config["optimizer"], model)
    lr_scheduler = get_lr_scheduler(config["lr_scheduler"], optimizer)
    
    
    loss = get_loss(config["loss"])

    # now create train object



if __name__ == "__main__":
    main(CONFIG_PATH)
