import json
import logging
import os

def load_config(config_path="config.json"):
    """Loads configuration from a JSON file or uses default values if the file is not found."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {
            "supported_extensions": [".txt", ".docx", ".pdf", ".doc"],
            "analysis_max_tokens": 8192,
            "nomenclature_max_tokens": 8192,
            "analysis_temperature": 0.7,
            "analysis_top_p": 0.8,
            "nomenclature_temperature": 0.2,
            "nomenclature_top_p": 0.5
        }
    return config


# Load configuration or use defaults
config = load_config()

# Extract specific configuration values
supported_ext = tuple(config["supported_extensions"])
analysis_max_tokens = config["analysis_max_tokens"]
nomenclature_max_tokens = config["nomenclature_max_tokens"]
analysis_temperature = config["analysis_temperature"]
analysis_top_p = config["analysis_top_p"]
nomenclature_temperature = config["nomenclature_temperature"]
nomenclature_top_p = config["nomenclature_top_p"]

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