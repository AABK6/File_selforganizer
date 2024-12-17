import os
import shutil
from config import logger

def organize_files(base_dir, folder_structure):
    """
    Organizes files into folders based on the provided structure.
    
    Args:
        base_dir (str): The base directory where files are located.
        folder_structure (dict): A dictionary representing the folder structure.
    """
    logger.info("Organizing files...")

    for filename, category_path in folder_structure.items():
        source_path = os.path.join(base_dir, filename)
        destination_path = os.path.join(base_dir, category_path)

        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        try:
            shutil.move(source_path, destination_path)
            logger.info(f"Moved '{filename}' to '{category_path}'")
        except FileNotFoundError:
            logger.warning(f"File not found: '{filename}'")
        except Exception as e:
            logger.error(f"Error moving '{filename}': {e}")

def format_structure_output(structure):
    """Formats the proposed folder structure for better readability."""

    def format_recursive(struct, indent=0):
        output = ""
        for key, value in struct.items():
            output += "  " * indent + "- " + key + "\n"
            if isinstance(value, dict):
                output += format_recursive(value, indent + 1)
        return output

    return format_recursive(structure)

def create_folders_recursively(base_dir, folder_structure):
    """
    Recursively creates folders based on the provided folder structure.

    Args:
        base_dir (str): The base directory where folders will be created.
        folder_structure (dict): A dictionary representing the folder structure.
    """
    for folder_path in set(folder_structure.values()):  # Use set to avoid creating duplicates
        full_path = os.path.join(base_dir, folder_path)
        
        try:
            os.makedirs(full_path, exist_ok=True)  # exist_ok=True prevents errors if folder exists
            logger.info(f"Created folder: {full_path}")
        except OSError as e:
            logger.error(f"Error creating folder '{full_path}': {e}")




