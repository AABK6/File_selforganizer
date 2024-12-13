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
import jsonschema
from jsonschema import validate

# External dependencies:
#   google-generativeai (LLM)
#   docx
#   PyPDF2

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-generativeai not found. Please install it.")
    sys.exit(1)

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
client = genai.Client(api_key=api_key)

analysis_response_schema = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {"type": "string"},
        "entities": {
            "type": "object",
            "properties": {
                "authors": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "intended_recipients": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "organizations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dates": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["authors", "intended_recipients"]
        },
        "key_phrases": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sentiment": {"type": "string"},
        "document_form": {"type": "string"},
        "document_purpose": {"type": "string"},
    },
    "required": ["tags", "summary", "entities", "key_phrases", "sentiment", "document_form", "document_purpose"]
}

analysis_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": analysis_max_tokens,
    "response_schema": analysis_response_schema,
    "response_mime_type": "application/json",
}

nomenclature_generation_config = {
    "temperature": 0.8,
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
        self.analysis_config = analysis_config
        self.nomenclature_config = nomenclature_config
    
    def analyze_text(self, text: str) -> dict:
        """Analyze a text using the analysis model."""
        prompt = (
            "You are an expert document analyzer. Your job is to analyze the following text and extract specified information and return it in strict JSON format. "
            "The output must be valid JSON with no commentary or explanations. No extra text should surround the JSON object. "
            "Ensure that all strings are properly terminated with double quotes.\n\n"
            "The JSON structure is as follows:\n"
            "{\n"
            '  "tags": ["tag1", "tag2", "tag3", ... ],\n'
            '  "summary": "A brief summary of the document.",\n'
            '  "entities": {\n'
            '    "authors": ["Author Name"],\n'
            '    "intended_recipients": ["Recipient Name"],\n'
            '    "organizations": ["Organization Name"],\n'
            '    "locations": ["Location Name"],\n'
            '    "dates": ["2024-01-01"]\n'
            '  },\n'
            '  "key_phrases": ["phrase1", "phrase2", ... ],\n'
            '  "sentiment": "neutral",\n'
            '  "document_form": "report",\n'
            '  "document_purpose": "informative"\n'
            '}\n\n'
            "Now, analyze the following text:\n"
        )
    
        full_prompt = prompt + text
        response = client.models.generate_content(
            model='gemini-1.5-flash-8b',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                **self.analysis_config
            )
        )

        if response and response.text:
            try:
                # Attempt JSON normalization
                json_string = response.text.strip()
                json_string = json_string.replace('\\', '\\\\')  # Escape backslashes
                json_string = json_string.replace('\n', '')  # Remove newlines
                json_string = json_string.replace('\'', '"')  # Replace single quotes

                # Attempt JSON parsing
                result = json.loads(json_string)
                 # Validate the response against the updated schema
                validate(instance=result, schema=analysis_response_schema)

                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError during LLM analysis: {e}")
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"ValidationError during LLM analysis: {e}")
            except Exception as e:
                logger.error(f"General exception during LLM analysis: {e}")
        return {
            'tags': [],
            'summary': 'No summary available.',
            'entities': {},
            'key_phrases': [],
            'sentiment': 'neutral',
            'document_form': '',
            'document_purpose': ''
        }

    def propose_structure(self, analysis_results: list[dict], user_feedback: str) -> dict:
        """Propose a folder structure using the nomenclature model."""
        # Reworked prompt: instruct the model to reason about flat vs hierarchical
        # and consider user feedback. No commentary, just return JSON.
        prompt = (
            "You have a list of files with detailed metadata including tags, summary, entities, key phrases, sentiment, document form, and document purpose. "
            "Your job: reorganize the files based on their content and propose a neat and logical folder structure. "
            "Group files by their thematic similarities, key phrases, and as a possible second layer on document form/purpose. "
            "You may use a flat structure or a hierarchical folder structure. If some groups are subsets of others, create nested folders. If not, keep it simpler. "
            "Consider this user feedback to improve your proposal:\n"
            f"{user_feedback}\n\n"
            "The files are:\n\n"
            + json.dumps(analysis_results, indent=2) + "\n\n"
            "Return only a JSON object: folder names as keys, and values either arrays of file paths "
            "or nested objects for subfolders. No extra text."
        )

        response = client.models.generate_content(
          model='gemini-1.5-flash-8b',
           contents=prompt,
          config=types.GenerateContentConfig(
                **self.nomenclature_config
            )
        )
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
        current_hash = get_file_hash(fpath)
        cached_data = None if self.force_reanalyze else get_cached_analysis(current_hash)
        
        # Use cached result if file unchanged and reanalyze not forced
        if cached_data:
            return cached_data.get("analysis", {})

        # Extract text and analyze via LLM
        text = extract_file_content(fpath)
        if not text.strip():
            analysis = {
                'tags': [],
                'summary': 'No summary available.',
                'entities': {
                    'authors': [],
                    'intended_recipients': [],
                    'organizations': [],
                    'locations': [],
                    'dates': []
                },
                'key_phrases': [],
                'sentiment': 'neutral',
                'document_form': '',
                'document_purpose': ''
            }
        else:
            analysis = self.llm_client.analyze_text(text)

        # Include the filepath in the analysis result
        combined_result = {
            "filepath": fpath,
            **analysis
        }

        # Save analysis for future use
        save_analysis(current_hash, analysis)
        return combined_result
       

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

def gather_user_feedback_and_improve(llm_client: LLMClient, analysis_results: list[dict], output_dir: str):
    """Loop until user approves structure or quits. On 'c', get user feedback and re-propose."""
    user_feedback = ""
    while True:
        proposed_structure = llm_client.propose_structure(analysis_results, user_feedback)
        logger.info("Proposed structure:")
        formatted_structure = format_structure_output(proposed_structure)
        print("\nProposed structure:\n")
        print(formatted_structure)

        # Aggregate and display key metadata for user awareness
        all_authors = set(author for item in analysis_results for author in item.get("entities", {}).get("authors", []))
        all_recipients = set(recipient for item in analysis_results for recipient in item.get("entities", {}).get("intended_recipients", []))
        all_key_phrases = set(phrase for item in analysis_results for phrase in item.get("key_phrases", []))
        overall_sentiment = [item.get("sentiment", "neutral") for item in analysis_results]
        sentiment_counts = {}
        for sentiment in overall_sentiment:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logger.info(f"Identified Authors: {', '.join(all_authors) if all_authors else 'None'}")
        logger.info(f"Identified Intended Recipients: {', '.join(all_recipients) if all_recipients else 'None'}")
        logger.info(f"Identified Key Phrases: {', '.join(all_key_phrases) if all_key_phrases else 'None'}")
        logger.info(f"Sentiment Distribution: {sentiment_counts}")

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
        report_data = {
            "analysis_results": [
                {
                    "filepath": item.get("filepath", ""),
                    "tags": item["analysis"].get("tags", []),
                    "summary": item["analysis"].get("summary", ""),
                    "entities": item["analysis"].get("entities", {}),
                    "key_phrases": item["analysis"].get("key_phrases", []),
                    "sentiment": item["analysis"].get("sentiment", ""),
                    "document_form": item["analysis"].get("document_form", ""),
                    "document_purpose": item["analysis"].get("document_purpose", "")
                } for item in analysis_results
            ]
        }
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

    Path(output_dir).mkdir(parents=True, exist_ok=True) # Ensure output directory is made

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
