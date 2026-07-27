import yaml
from pathlib import Path

def load_config_yaml(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)
    
def save_config_yaml(odirname, config):
    odirname = Path(odirname)
    odirname.mkdir(parents=True, exist_ok=True)
    with open(f"{odirname}/config.yml", "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)