


from NeuralNetworkSegmentation.optimizer import get_optimizer
from NeuralNetworkSegmentation.models.unet import get_model
from NeuralNetworkSegmentation.config import load_config_yaml


def main(
        config_path,
):
    
    config = load_config_yaml(config_path)

    model = get_model(config["model"])
    optimizer = get_optimizer(config["optimizer"], model)

    
    print(model)
    print(config)

