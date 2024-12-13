import os
import json
import shutil
import sys
import subprocess
import uuid
import hashlib
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import re

# External dependencies:
#   google-generativeai (LLM)
#   docx
#   PyPDF2

try:
    import docx
except ImportError:
    print("docx library not found. Please install python-docx.")
    sys.exit(1)
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 library not found. Please install PyPDF2.")
    sys.exit(1)
try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import content
except ImportError:
    print("google-generativeai not found. Please install it.")
    sys.exit(1)


########################################################################
# Logging
########################################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("organizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


########################################################################
# Configuration
########################################################################

def load_config(config_path="config.json"):
    """Load configuration from JSON file or use defaults if not found."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("Config file not found. Using defaults.")
        return {
            "supported_extensions": ['.txt', '.md', '.doc', '.docx', '.pdf'],
            "analysis_max_tokens": 8000,
            "nomenclature_max_tokens": 4000,
        }


config = load_config()
supported_ext = tuple(config.get("supported_extensions", ['.txt', '.md', '.doc', '.docx', '.pdf']))
analysis_max_tokens = config.get("analysis_max_tokens", 8000)
nomenclature_max_tokens = config.get("nomenclature_max_tokens", 4000)

########################################################################
# LLM Initialization
########################################################################

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not set. Exiting.")
    sys.exit(1)

genai.configure(api_key=api_key)

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

analysis_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": analysis_max_tokens,
    "response_schema": analysis_response_schema,
    "response_mime_type": "application/json",
}

nomenclature_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": nomenclature_max_tokens,
    "response_mime_type": "application/json",
}


########################################################################
# Storage and Caching
########################################################################

STORAGE_FILE = "analysis_cache.json"

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

def get_file_id(filepath: str) -> str:
    """Generate a unique file ID based on the filepath."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))

def get_cached_analysis(file_id: str) -> dict:
    """Get cached analysis for a file if it exists."""
    storage = load_storage()
    return storage["files"].get(file_id)

def save_analysis(file_id: str, analysis: dict, file_hash: str) -> None:
    """Save analysis to cache."""
    storage = load_storage()
    storage["files"][file_id] = {"analysis": analysis, "hash": file_hash}
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


########################################################################
# File Extraction Functions
########################################################################

def extract_text_from_txt(filepath: str) -> str:
    """Extract text from a .txt or .md file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except:
        logger.warning(f"Failed to read {filepath}")
        return ""

def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a .docx file."""
    try:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs)
    except:
        logger.warning(f"Failed to extract text from {filepath}")
        return ""

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a .pdf file."""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                t = page.extract_text() or ""
                pages.append(t)
            return "\n".join(pages)
    except:
        logger.warning(f"Failed to extract text from {filepath}")
        return ""

def extract_text_from_doc(filepath: str) -> str:
    """Extract text from a .doc file using antiword."""
    try:
        result = subprocess.run(["antiword", filepath], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning(f"antiword failed for {filepath}")
            return ""
    except FileNotFoundError:
        logger.warning("antiword not found. Skipping .doc extraction.")
        return ""

def extract_file_content(filepath: str) -> str:
    """Dispatch extraction based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext in ['.txt', '.md']:
        return extract_text_from_txt(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    elif ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.doc':
        return extract_text_from_doc(filepath)
    return ""


########################################################################
# Hashing
########################################################################

def get_file_hash(filepath: str) -> str:
    """Compute a SHA256 hash for a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return ""
    return hasher.hexdigest()


########################################################################
# LLM Client Classes
########################################################################

class LLMClient:
    """A simple client for interacting with the LLM models."""
    def __init__(self, analysis_config, nomenclature_config):
        self.analysis_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config=analysis_config
        )
        self.nomenclature_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            generation_config=nomenclature_config
        )
    
    def analyze_text(self, text: str) -> dict:
        """Analyze a text using the analysis model."""
        chat_session = self.analysis_model.start_chat()
        response = chat_session.send_message(text)
        if response and response.text:
            try:
                result = json.loads(response.text)
                return {
                    'tags': result.get('tags', []),
                    'summary': result.get('summary', '')
                }
            except ValueError:
                logger.error("Error parsing JSON from LLM analysis.")
        return {'tags': [], 'summary': 'No summary available.'}

    def propose_structure(self, analysis_results: list[dict], user_feedback: str) -> dict:
        """Propose a folder structure using the nomenclature model."""
        # Reworked prompt: instruct the model to reason about flat vs hierarchical
        # and consider user feedback. No commentary, just return JSON.
        prompt = (
            "You have a list of files with tags and summaries. "
            "Your job: reorganize the files based on their content and propose a neat and logical folder structure. "
            "Group files by their thematic similarities. "
            "You may use a flat structure or a hierarchical folder structure. If some groups are subsets of others, create nested folders. If not, keep it simpler. "
            "Consider this user feedback to improve your proposal:\n"
            f"{user_feedback}\n\n"
            "The files are:\n\n"
            + json.dumps(analysis_results, indent=2) + "\n\n"
            "Return only a JSON object: folder names as keys, and values either arrays of file paths "
            "or nested objects for subfolders. No extra text."
        )

        chat_session = self.nomenclature_model.start_chat()
        response = chat_session.send_message(prompt)
        if response and response.text:
            try:
                return json.loads(response.text)
            except ValueError:
                logger.error("Error parsing JSON from LLM for nomenclature proposal.")
        else:
            logger.error("No response from LLM for nomenclature proposal.")
        return {"Misc": [item["filepath"] for item in analysis_results]}


########################################################################
# File Analysis Class
########################################################################

class FileAnalyzer:
    """Handles file scanning, content extraction, and analysis caching."""
    def __init__(self, llm_client: LLMClient, force_reanalyze: bool):
        self.llm_client = llm_client
        self.force_reanalyze = force_reanalyze

    def process_file(self, fpath: str) -> dict:
        """Process a single file: extract, analyze, cache result."""
        file_id = get_file_id(fpath)
        cached_data = None if self.force_reanalyze else get_cached_analysis(file_id)
        current_hash = get_file_hash(fpath)
        
        # Use cached result if file unchanged and reanalyze not forced
        if cached_data and cached_data.get("hash") == current_hash:
            return cached_data.get("analysis", {})

        # Extract text and analyze via LLM
        text = extract_file_content(fpath)
        if not text.strip():
            analysis = {'tags': [], 'summary': 'No extractable text.'}
        else:
            analysis = self.llm_client.analyze_text(text)

        # Save analysis for future use
        save_analysis(file_id, analysis, current_hash)
        return analysis

    def process_directory(self, input_dir: str) -> list[dict]:
        """Scan directory, process supported files, return analysis results."""
        files = self.scan_directory(input_dir)
        analysis_results = []
        for fpath in tqdm(files, desc="Analyzing files"):
            analysis = self.process_file(fpath)
            analysis_results.append({"filepath": fpath, "analysis": analysis})
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

def gather_user_feedback_and_improve(llm_client: LLMClient, analysis_results: list[dict], output_dir: str):
    """Loop until user approves structure or quits. On 'c', get user feedback and re-propose."""
    user_feedback = ""
    while True:
        proposed_structure = llm_client.propose_structure(analysis_results, user_feedback)
        logger.info("Proposed structure:")
        logger.info(json.dumps(proposed_structure, indent=2))

        choice = input("Approve (a), Reject (r), or Comment (c)? ").strip().lower()
        if choice == 'a':
            # Approved
            user_comment = input("Optional comment about the final structure (press Enter to skip): ")
            save_nomenclature_comment(proposed_structure, user_comment)
            organize_files(proposed_structure, output_dir)
            generate_report(analysis_results, proposed_structure, output_dir)
            logger.info("Files organized.")
            break
        elif choice == 'r':
            # Rejected with no feedback
            logger.info("User rejected the structure. Trying again without additional feedback.")
            user_feedback = ""
        elif choice == 'c':
            # User wants to refine structure
            user_feedback = input("Enter your feedback to improve the structure: ")
            logger.info("Will re-propose structure with user feedback.")
        else:
            logger.info("Invalid choice. Please enter 'a', 'r', or 'c'.")


def generate_report(analysis_results: list[dict], proposed_structure: dict, output_dir: str):
    """Generate a report JSON file of the final analysis and structure."""
    report_path = Path(output_dir) / "report.json"
    with report_path.open('w', encoding='utf-8') as rep:
        report_data = {
            "analysis_results": [
                {
                    "filepath": item["filepath"],
                    "analysis": item["analysis"],
                    "file_id": get_file_id(item["filepath"])
                } for item in analysis_results
            ],
            "final_structure": proposed_structure
        }
        report_data.update({"nomenclature_comment": get_nomenclature_comment()})
        json.dump(report_data, rep, indent=2)
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

    # Initialize LLM client and file analyzer
    llm_client = LLMClient(analysis_generation_config, nomenclature_generation_config)
    analyzer = FileAnalyzer(llm_client, args.force_reanalyze)

    # Analyze all files
    analysis_results = analyzer.process_directory(input_dir)
    if not analysis_results:
        logger.info("No supported files found.")
        return

    # Prompt user to approve structure, reject, or provide feedback for refinement
    gather_user_feedback_and_improve(llm_client, analysis_results, output_dir)


if __name__ == "__main__":
    main()




