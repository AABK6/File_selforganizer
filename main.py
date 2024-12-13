import os
import json
import shutil
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from pathlib import Path
import sys
import docx
import PyPDF2
import subprocess
import uuid
import hashlib
import logging
import argparse
from tqdm import tqdm
import re


# --- Configuration Loading ---
def load_config(config_path="config.json"):
    """Loads the configuration from the JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Config file not found. Using default values.")
        return {
            "supported_extensions": ['.txt', '.md', '.doc', '.docx', '.pdf'],
            "analysis_max_tokens": 8000,
            "nomenclature_max_tokens": 4000,
        }

config = load_config()
supported_ext = tuple(config.get("supported_extensions", ['.txt', '.md', '.doc', '.docx', '.pdf']))
analysis_max_tokens = config.get("analysis_max_tokens", 8000)
nomenclature_max_tokens = config.get("nomenclature_max_tokens", 4000)

# --- End Configuration ---

# configure the API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set. Exiting.")
    sys.exit(1)
genai.configure(api_key=api_key)


# Define the response schema
analysis_response_schema = content.Schema(
    type=content.Type.OBJECT,
    properties={
        "tags": content.Schema(
            type=content.Type.ARRAY,
            items=content.Schema(type=content.Type.STRING)
        ),
        "summary": content.Schema(type=content.Type.STRING),
    },
    required=["tags", "summary"]
)


# Define the generation configuration
analysis_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": analysis_max_tokens,
    "response_schema": analysis_response_schema,
    "response_mime_type": "application/json",
}


# Create the model
analysis_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=analysis_generation_config,
)


nomenclature_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": nomenclature_max_tokens,
    "response_mime_type": "application/json",
}

nomenclature_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=nomenclature_generation_config,
)


def extract_text_from_docx(filepath: str) -> str:
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_pdf(filepath: str) -> str:
    text = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def extract_text_from_doc(filepath: str) -> str:
    try:
        result = subprocess.run(["antiword", filepath], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning(f"Could not extract text from {filepath} using antiword.")
    except FileNotFoundError:
        logger.warning("antiword not found. Unable to extract text from .doc file.")
    return ""

def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_file_content(filepath: str) -> str:
    lower_ext = Path(filepath).suffix.lower()
    if lower_ext in ['.txt', '.md']:
        return extract_text_from_txt(filepath)
    elif lower_ext == '.docx':
        return extract_text_from_docx(filepath)
    elif lower_ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif lower_ext == '.doc':
        return extract_text_from_doc(filepath)
    else:
        return ""

def get_file_hash(filepath: str) -> str:
    """Computes the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while True:
                chunk = file.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return None
    return hasher.hexdigest()


def analyze_file_content(filepath: str, force_analyze: bool = False) -> dict:
    """
    Analyzes file content and provides tags and a summary.
    If force_analyze is True, ignores cached data.
    """
    # Check if data is already cached and force_analyze is False
    file_id = get_file_id(filepath)
    if not force_analyze:
        cached_data = get_cached_analysis(file_id)
        if cached_data:
            logger.info(f"Using cached analysis for: {filepath}")
            file_hash = get_file_hash(filepath)
            if cached_data.get("hash") == file_hash:
                return cached_data.get("analysis", {})
            else:
                 logger.info(f"File changed: {filepath}, reanalyzing...")

    # If no cached data or force_analyze is True, analyze the file
    file_data = extract_file_content(filepath)

    if not file_data.strip():
        # If no text extracted, provide fallback analysis
        analysis = {
            'tags': [],
            'summary': 'No extractable text.'
        }
    else:
        chat_session = analysis_model.start_chat()
        response = chat_session.send_message(file_data)
        if response and response.text:
            try:
                result = json.loads(response.text)
                analysis = {
                    'tags': result.get('tags', []),
                    'summary': result.get('summary', '')
                }
            except ValueError:
                logger.error(f"Error parsing JSON response for file: {filepath}")
                analysis = {
                    'tags': [],
                    'summary': 'No summary available.'
                }
        else:
            logger.error(f"No response from LLM for file: {filepath}")
            analysis = {
                    'tags': [],
                    'summary': 'No summary available.'
                }
    # Save the analysis and return
    file_hash = get_file_hash(filepath)
    save_analysis(file_id, analysis, file_hash)
    return analysis


def propose_nomenclature_with_llm(analysis_results: list[dict]) -> dict:
    """
    Sends the entire analysis to the LLM and asks it to propose a nomenclature.
    """
    # Prepare a prompt that gives the LLM instructions and the entire analysis
    prompt = (
        "You are given a list of files and their extracted tags and summaries. "
        "Your task is to propose a meaningful folder structure (a JSON object) "
        "that groups the files by their thematic similarity. Each key should be "
        "the name of a folder, and the value an array of file paths. Make sure "
        "to use descriptive folder names that reflect the themes emerging from "
        "the tags and summaries, and group similar files together. "
        "Only return valid JSON with no additional commentary.\n\n"
        "Here is the data:\n\n"
        + json.dumps(analysis_results, indent=2) + "\n\n"
        "Now respond with the proposed folder structure."
    )

    chat_session = nomenclature_model.start_chat()
    response = chat_session.send_message(prompt)
    if response and response.text:
        try:
            result = json.loads(response.text)
            return result
        except ValueError:
            logger.error("Error parsing JSON response for nomenclature proposal.")
    else:
        logger.error("No response from LLM for nomenclature proposal.")

    # Fallback if no valid response
    return {"Misc": [item["filepath"] for item in analysis_results]}


def organize_files(chosen_structure: dict, output_dir: str) -> None:
    for theme, files in chosen_structure.items():
        theme_dir = Path(output_dir) / Path(theme.strip().replace(" ", "_"))
        if not theme_dir.exists():
            theme_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            basename = os.path.basename(f)
            src = Path(f)
            dest = theme_dir / basename
            if src.exists():
                shutil.move(str(src), str(dest))
                logger.info(f"Moved {str(src)} to {str(dest)}")
            else:
                logger.warning(f"File {f} does not exist and cannot be moved.")


# --- Persistence functions ---

STORAGE_FILE = "analysis_cache.json"

def load_storage() -> dict:
    """Loads the persisted analysis data from file."""
    if Path(STORAGE_FILE).exists():
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "nomenclature_comments": {}}


def save_storage(data: dict) -> None:
    """Saves the analysis data to the storage file."""
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_file_id(filepath: str) -> str:
    """Generates a unique ID based on file path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))  # Use filepath to generate a UUID


def get_cached_analysis(file_id: str) -> dict or None:
    """Retrieves the analysis result for a given file."""
    storage = load_storage()
    return storage["files"].get(file_id)


def save_analysis(file_id: str, analysis: dict, file_hash:str) -> None:
    """Stores or updates the analysis data."""
    storage = load_storage()
    if "files" not in storage:
        storage["files"] = {}
    storage["files"][file_id] = {"analysis": analysis, "hash": file_hash}
    save_storage(storage)


def save_nomenclature_comment(proposed_structure: dict, comment: str):
    """Stores the user comment related to the nomenclature proposal."""
    storage = load_storage()
    if "nomenclature_comments" not in storage:
       storage["nomenclature_comments"]={}
    storage["nomenclature_comments"] = {"structure":proposed_structure, "comment": comment}
    save_storage(storage)

def get_nomenclature_comment():
    """Retrieves the comment related to the proposed structure."""
    storage = load_storage()
    return storage.get("nomenclature_comments",{})

# --- End of Persistence ---

# --- Logger ---
def create_logger():
    """Configures and returns a logger object."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("organizer.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# --- End of Logger ---

def scan_directory(directory: str, supported_ext: tuple) -> list[str]:
    """Recursively scans a directory for supported files."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(supported_ext):
                files.append(str(Path(root) / filename))
    return files


def main():
    """
    Main function orchestrates the workflow:
    1. Loads configurations.
    2. Sets up the logger.
    3. Takes input/output paths and scan the folder.
    4. Analyzes file content using the Gemini API (or cached data if available).
    5. Proposes a folder structure.
    6. Asks the user to validate the structure and add a comment.
    7. Organizes files and generates a report.
    """
    global logger
    logger = create_logger()

    parser = argparse.ArgumentParser(description="Organize files using LLM.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory (default: 'organized_folder' in the input directory).",
    )
    parser.add_argument(
         "--force_reanalyze",
         action="store_true",
         help="Force re-analysis of all files, bypassing the cache."
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    force_reanalyze = args.force_reanalyze

    if not os.path.exists(input_dir):
            logger.error(f"Input path {input_dir} does not exist. Exiting.")
            sys.exit(1)

    # Set output dir
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_dir), 'organized_folder')
    logger.info(f"Output path set to: {output_dir}")

    files_to_process = scan_directory(input_dir, supported_ext)

    if not files_to_process:
        logger.info("No supported files found.")
        return

    # Analyze files
    analysis_results = []
    for fpath in tqdm(files_to_process, desc="Analyzing files"):
        analysis = analyze_file_content(fpath, force_analyze=force_reanalyze)
        analysis_results.append({"filepath": fpath, "analysis": analysis})

    # Propose nomenclature with LLM
    proposed_structure = propose_nomenclature_with_llm(analysis_results)
    logger.info("Proposed Structure:")
    logger.info(json.dumps(proposed_structure, indent=2))

    # Validate with the user and save the comment
    user_input = input("Approve this structure? (y/n): ").strip().lower()
    if user_input != 'y':
        logger.info("Structure not approved. Exiting.")
        return
    user_comment = input("Enter your comment about the structure (or press Enter to skip): ")
    save_nomenclature_comment(proposed_structure, user_comment)

    # Organize files
    organize_files(proposed_structure, output_dir)

    # Generate a report
    report_path = Path(output_dir) / "report.json"
    with report_path.open('w', encoding='utf-8') as rep:
        report_data = {
            "analysis_results": [
                {
                    "filepath": item["filepath"],
                    "analysis": item["analysis"],
                    "file_id": get_file_id(item["filepath"]),
                } for item in analysis_results
            ],
            "final_structure": proposed_structure,
        }
        report_data.update({"nomenclature_comment": get_nomenclature_comment()})
        json.dump(report_data, rep, indent=2)

    logger.info(f"Report generated: {report_path}")
    logger.info("Files organized successfully.")


if __name__ == "__main__":
    main()