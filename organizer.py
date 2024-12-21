import os
import shutil
from config import logger

def organize_files(base_dir, folder_structure, storage):
    """
    Organizes files into folders based on the provided structure.

    Args:
        base_dir (str): The base directory where files are located.
        folder_structure (dict): A dictionary representing the folder structure.
        storage (dict): The storage dictionary containing file information.
    """
    logger.info("Organizing files...")

    for file_hash, file_info in storage.items():
        filename = os.path.basename(file_info['path'])
        analysis = file_info.get('analysis', {})  # Get analysis, default to empty dict

        # Find the correct destination path based on the approved structure
        destination_folder = find_destination_folder(folder_structure, analysis)
        if destination_folder:
            source_path = file_info['path']
            destination_path = os.path.join(base_dir, destination_folder, filename)

            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            try:
                shutil.move(source_path, destination_path)
                logger.info(f"Moved '{filename}' to '{os.path.join(destination_folder, filename)}'")
            except FileNotFoundError:
                logger.warning(f"File not found: '{filename}'")
            except Exception as e:
                logger.error(f"Error moving '{filename}': {e}")
        else:
            logger.warning(f"No destination folder found for '{filename}'.")

def find_destination_folder(folder_structure, analysis_data):
    """
    Finds the destination folder path within the nested structure based on analysis data.
    This is a placeholder and needs to be implemented based on how your LLM proposes the structure
    and how you want to map analysis data to folder paths.
    """
    # Simple example: Assuming the top-level keys in folder_structure are categories
    for category, sub_structure in folder_structure.items():
        # Implement your logic to determine if the analysis_data belongs to this category
        # For example, check if a specific tag or entity is present in the analysis_data
        if analysis_data and analysis_data.get('tags') and category in analysis_data['tags']:
            return category
        # Add more sophisticated logic to traverse sub_structures if needed
    return None

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
    def create_folders(structure, parent_dir):
        for folder_name, sub_structure in structure.items():
            folder_path = os.path.join(parent_dir, folder_name)
            try:
                os.makedirs(folder_path, exist_ok=True)
                logger.info(f"Created folder: {folder_path}")
                if isinstance(sub_structure, dict):
                    create_folders(sub_structure, folder_path)
            except OSError as e:
                logger.error(f"Error creating folder '{folder_path}': {e}")

    create_folders(folder_structure, base_dir)