import json
import logging
import os

# Load configuration or use defaults
config = load_config()

# Extract specific configuration values
supported_ext = tuple(config["supported_extensions"])
analysis_max_tokens = config["analysis_max_tokens"]
nomenclature_max_tokens = config["nomenclature_max_tokens"]

def load_config(config_path="config.json"):
    """Loads configuration from a JSON file or uses default values if the file is not found."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {
            "supported_extensions": [".txt", ".docx", ".pdf", ".doc"],
            "analysis_max_tokens": 1500,
            "nomenclature_max_tokens": 1000,
        }
    return config

# Logger setup
logger = logging.getLogger("organizer")
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("organizer.log")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)