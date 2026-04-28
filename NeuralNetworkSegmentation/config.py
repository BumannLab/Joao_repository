import yaml

def load_config_yaml(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)