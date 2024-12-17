import os
import json
from pathlib import Path

from file_utils import extract_file_content, get_file_hash
from llm_client import LLMClient, analysis_generation_config, nomenclature_generation_config
from organizer import organize_files, create_folders_recursively, format_structure_output
from config import logger, config, supported_ext
from cli import gather_user_feedback_and_improve, parse_arguments


def load_storage(storage_file="storage.json"):
    """Loads the storage file, or creates it if it doesn't exist."""
    try:
        with open(storage_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_storage(storage, storage_file="storage.json"):
    """Saves the storage to a JSON file."""
    with open(storage_file, "w") as f:
        json.dump(storage, f, indent=4)


class FileAnalyzer:
    def __init__(self, llm_client, storage):
        self.llm_client = llm_client
        self.storage = storage

    def process_file(self, file_path):
        """Processes a single file, extracting content and analyzing it with the LLM."""
        file_hash = get_file_hash(file_path)
        if file_hash in self.storage:
            logger.info(f"File {file_path} already processed, skipping.")
            return self.storage[file_hash]

        file_content = extract_file_content(file_path)
        analysis_result = self.llm_client.analyze_text(file_content)
        self.storage[file_hash] = {
            "path": str(file_path),
            "content": file_content,
            "analysis": analysis_result,
        }
        return analysis_result

    def process_directory(self, directory):
        """Processes all supported files in a directory."""
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file() and file_path.suffix in supported_ext:
                self.process_file(file_path)

    def scan_directory(self, directory):
        """Scans the directory and returns a list of supported files"""
        files = []
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file() and file_path.suffix in supported_ext:
                files.append(file_path)
        return files


def generate_report(storage, report_file="report.json"):
    """Generates a JSON report of the file analysis results."""
    with open(report_file, "w") as f:
        json.dump(storage, f, indent=4)


def main():
    """Main function to orchestrate the file organization process."""
    args = parse_arguments()

    llm_client = LLMClient(analysis_generation_config, nomenclature_generation_config)
    storage = load_storage()

    file_analyzer = FileAnalyzer(llm_client, storage)

    files = file_analyzer.scan_directory(args.directory)
    for file in files:
        file_analyzer.process_file(file)

    proposed_structure = llm_client.propose_structure(
        [item["analysis"] for item in storage.values()]
    )

    formatted_structure = format_structure_output(proposed_structure)
    print("Proposed folder structure:")
    print(formatted_structure)

    approved_structure = gather_user_feedback_and_improve(
        llm_client, proposed_structure
    )

    if approved_structure:
        logger.info("Organizing files...")
        create_folders_recursively(args.directory, approved_structure)
        organize_files(args.directory, approved_structure, storage)
        logger.info("Files organized successfully!")
    else:
        logger.info("File organization cancelled.")

    save_storage(storage)
    generate_report(storage)


if __name__ == "__main__":
    main()