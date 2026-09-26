# src/config.py
import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Reads the centralized YAML rulebook and returns it as a Python dictionary.
    """
    # Verify the file exists before trying to open it
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing rulebook! Could not find {config_path}")
        
    with open(config_path, "r") as file:
        return yaml.safe_load(file)