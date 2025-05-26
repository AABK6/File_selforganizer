import os
import json
import shutil
import sys
# import subprocess # Moved to utils.py
import uuid
# import hashlib # Moved to utils.py
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from llm_client2 import LLMClient as LLMClient2, llm_config as llm_client2_config # Updated import
from .utils import extract_file_content, get_file_hash # Added import

# External dependencies:
#   google-generativeai (LLM)
#   docx (now in utils.py)
#   PyPDF2 (now in utils.py)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-generativeai not found. Please install it.")
    sys.exit(1)

# Removed docx and PyPDF2 try-except blocks as they are handled in utils.py
# try:
#     import docx
# except ImportError:
#     print("docx library not found. Please install python-docx.")
#     sys.exit(1)
# try:
#     import PyPDF2
# except ImportError:
#     print("PyPDF2 library not found. Please install PyPDF2.")
#     sys.exit(1)


########################################################################
# Logging
########################################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "organizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


########################################################################
# Configuration
########################################################################

def load_config(config_path=None):
    """Load configuration from JSON file or use defaults if not found."""
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("Config file not found. Using defaults.")
        return {
            "supported_extensions": ['.txt', '.md', '.doc', '.docx', '.pdf'],
            "analysis_max_tokens": 8192,
            "nomenclature_max_tokens": 8192,
        }


config = load_config()
supported_ext = tuple(config.get("supported_extensions", ['.txt', '.md', '.doc', '.docx', '.pdf']))
analysis_max_tokens = config.get("analysis_max_tokens", 8192)
nomenclature_max_tokens = config.get("nomenclature_max_tokens", 8192)

########################################################################
# LLM Initialization
########################################################################

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not set. Exiting.")
    sys.exit(1)

# Only run this block for Google AI API
client = genai.Client(api_key=api_key) # This is the main genai.Client
gemini_models_client = client.models # This is what LLMClient2 expects

# analysis_response_schema, analysis_generation_config, and nomenclature_generation_config
# have been removed as they are now handled by llm_client2.py

########################################################################
# Storage and Caching
########################################################################

script_dir = Path(__file__).parent
STORAGE_FILE = script_dir.parent / "analysis_cache.json"

def load_storage() -> dict:
    """Load analysis and nomenclature data from storage."""
    if Path(STORAGE_FILE).exists():
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "nomenclature_comments": {}}

def save_storage(data: dict) -> None:
    """Save analysis and nomenclature data to storage."""
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_cached_analysis(file_hash: str) -> dict:
    """Get cached analysis for a file if it exists."""
    storage = load_storage()
    return storage["files"].get(file_hash)

def save_analysis(file_hash: str, analysis: dict) -> None:
    """Save analysis to cache."""
    storage = load_storage()
    storage["files"][file_hash] = {"analysis": analysis}
    save_storage(storage)

def save_nomenclature_comment(proposed_structure: dict, comment: str):
    """Save user comment related to the proposed nomenclature."""
    storage = load_storage()
    storage["nomenclature_comments"] = {"structure": proposed_structure, "comment": comment}
    save_storage(storage)

def get_nomenclature_comment():
    """Get the user comment on the proposed nomenclature."""
    storage = load_storage()
    return storage.get("nomenclature_comments", {})


# File Extraction Functions and Hashing functions have been moved to src/utils.py
# and are imported at the top of this file.

########################################################################
# File Analysis Class
########################################################################

# The LLMClient class definition has been removed from here.
# It will be imported from llm_client2.py in the next step.

class FileAnalyzer:
    """Handles file scanning, content extraction, and analysis caching."""
    def __init__(self, llm_client: LLMClient2, force_reanalyze: bool): # Updated type hint
        self.llm_client = llm_client
        self.force_reanalyze = force_reanalyze

    def process_file(self, fpath: str) -> dict:
        """Process a single file: extract, analyze, cache result."""
        current_hash = get_file_hash(fpath)
        cached_data_wrapper = None if self.force_reanalyze else get_cached_analysis(current_hash)
        
        if cached_data_wrapper and "analysis" in cached_data_wrapper:
            cached_analysis = cached_data_wrapper["analysis"]
            # Assuming cached_analysis is already in the new list format.
            # If not, re-analysis will occur naturally.
            return {"filepath": fpath, "analysis": cached_analysis}

        text = extract_file_content(fpath)
        analysis_list = [] # Default for new structure (list of entities)
        if not text.strip():
            analysis_list = [] # No content, no entities
        else:
            try:
                # This now returns a list of entities from llm_client2.LLMClient
                analysis_list = self.llm_client.analyze_text(text)
            except Exception as e:
                logger.error(f"Error analyzing file {fpath}: {e}")
                analysis_list = [] # Set to empty list on error to maintain structure

        # Save the new analysis_list structure.
        # save_analysis expects the actual analysis data (the list in this case).
        save_analysis(current_hash, analysis_list) 
        
        return {"filepath": fpath, "analysis": analysis_list}
       

    def process_directory(self, input_dir: str) -> list[dict]:
        """Scan directory, process supported files, return analysis results."""
        files = self.scan_directory(input_dir)
        analysis_results = []
        for fpath in tqdm(files, desc="Analyzing files"):
            # process_file now returns a dict like {"filepath": fpath, "analysis": analysis_list}
            processed_file_data = self.process_file(fpath) 
            analysis_results.append(processed_file_data) # Append the whole dict
        return analysis_results

    @staticmethod
    def scan_directory(directory: str) -> list[str]:
        """Return a list of supported files in a directory."""
        found = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(supported_ext):
                    found.append(str(Path(root) / filename))
        return found


########################################################################
# File Organization
########################################################################

def organize_files(chosen_structure: dict, output_dir: str) -> None:
    """Recursively create folders and move files according to the proposed structure."""
    for theme, content_ in chosen_structure.items():
        theme_dir = Path(output_dir) / Path(theme.strip().replace(" ", "_"))
        theme_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(content_, list):
            # This folder holds files directly
            for f in content_:
                src = Path(f)
                if src.exists():
                    shutil.move(str(src), str(theme_dir / src.name))
                else:
                    logger.warning(f"{f} does not exist.")
        elif isinstance(content_, dict):
            # Nested folders
            organize_files(content_, str(theme_dir))
        else:
            logger.warning(f"Unexpected structure type for {theme}")


########################################################################
# User Interaction Flow
########################################################################

def format_structure_output(structure: dict, indent=0) -> str:
    """Formats the proposed structure for user presentation."""
    output = ""
    for key, value in structure.items():
        output += "  " * indent + f"- {key}:\n"
        if isinstance(value, dict):
            output += format_structure_output(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                 output += "  " * (indent+1) + f"  - {item}\n"
    return output

def gather_user_feedback_and_improve(llm_client: LLMClient2, analysis_results: list[dict], output_dir: str): # Updated type hint
    """Loop until user approves structure or quits. On 'c', get user feedback and re-propose."""
    user_feedback = ""
    while True:
        # Ensure analysis_results passed to propose_structure is compatible with llm_client2's expectations.
        # llm_client2.propose_structure expects a list of analysis results (which it gets).
        # The internal structure of each item in analysis_results is now:
        # {"filepath": "...", "analysis": [{'entity': ..., 'category': ...,}, ...]}
        # The prompt in llm_client2.propose_structure is:
        # "Analysis Results: {analysis_results}\n{history}"""
        # This means it will dump the whole new structure. This might be fine.
        proposed_structure = llm_client.propose_structure(analysis_results, user_feedback) # llm_client here should be the new client
        logger.info("Proposed structure:")
        formatted_structure = format_structure_output(proposed_structure)
        print("\nProposed structure:\n")
        print(formatted_structure)

        # Display new simplified metadata
        logger.info(f"Analyzed {len(analysis_results)} files.")
        unique_categories = set()
        for item in analysis_results:
            filepath = item.get("filepath", "Unknown filepath")
            analysis_list = item.get("analysis", [])
            entity_names = [entity.get("entity", "Unknown entity") for entity in analysis_list[:3]] # First 3 entities
            entities_summary = ", ".join(entity_names)
            if len(analysis_list) > 3:
                entities_summary += "..."
            logger.info(f"File: {filepath} - Contains {len(analysis_list)} entities (e.g., {entities_summary if entity_names else 'No entities found'}).")
            for entity in analysis_list:
                if entity.get("category"):
                    unique_categories.add(entity.get("category"))
        
        if unique_categories:
            logger.info(f"Identified Entity Categories: {', '.join(sorted(list(unique_categories)))}")
        else:
            logger.info("No entity categories identified across files.")

        choice = input("Approve (a), Reject (r), or Comment (c)? ").strip().lower()
        if choice == 'a':
            # Approved
            organize_files(proposed_structure, output_dir)
            generate_report(analysis_results, output_dir)
            logger.info("Files organized.")
            break
        elif choice == 'r':
            # Rejected with no feedback
            logger.info("User rejected the structure. Files not moved. Final report generated.")
            generate_report(analysis_results, output_dir)
            break
        elif choice == 'c':
            # User wants to refine structure
            user_feedback = input("Enter your feedback to improve the structure: ")
            logger.info("Will re-propose structure with user feedback.")
        else:
            logger.info("Invalid choice. Please enter 'a', 'r', or 'c'.")


def generate_report(analysis_results: list[dict], output_dir: str):
    # Generate a report JSON file of the final analysis.
    report_path = Path(output_dir) / "report.json"
    with report_path.open('w', encoding='utf-8') as rep:
        # Updated report_data to correctly save the new analysis_results structure
        report_data_items = []
        for item in analysis_results:
            report_data_items.append({
                "filepath": item.get("filepath", ""),
                "analysis": item.get("analysis", []) # This is the list of entity dicts
            })
        final_report_data = {"analysis_results": report_data_items}
        json.dump(final_report_data, rep, indent=2)
    logger.info(f"Report generated: {report_path}")


########################################################################
# Main Execution Flow
########################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Organize files using LLM.")
    parser.add_argument("input_dir", type=str, help="Path to input directory.")
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("--force_reanalyze", action="store_true", help="Force re-analysis of all files.")
    args = parser.parse_args()

    # Validate input directory
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        logger.error(f"Input path {input_dir} does not exist.")
        sys.exit(1)

    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_dir), 'organized_folder')
    logger.info(f"Output path: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True) # Ensure output directory is made

    # Initialize LLM client and file analyzer
    # LLMClient (old) and its configs (analysis_generation_config, nomenclature_generation_config) are removed.
    # Instantiate the new LLMClient2 from llm_client2.py
    llm_client_new = LLMClient2(gemini_models_client, llm_client2_config) # Use the new client
    analyzer = FileAnalyzer(llm_client_new, args.force_reanalyze) # Pass the new client

    # Analyze all files
    analysis_results = analyzer.process_directory(input_dir)
    if not analysis_results:
        logger.info("No supported files found.")
        return

    # Prompt user to approve structure, reject, or provide feedback for refinement
    gather_user_feedback_and_improve(llm_client_new, analysis_results, output_dir) # Pass the new client




if __name__ == "__main__":
    main()
