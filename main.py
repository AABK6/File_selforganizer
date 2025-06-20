import os
import json
import shutil
import sys
import subprocess
import uuid
import hashlib
import logging
import argparse
import time
import re
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import jsonschema
from jsonschema import validate
from colorama import Fore, Style, init as colorama_init

# External dependencies:
#   google-genai (LLM)
#   docx
#   PyPDF2

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-genai not found. Please install it.")
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
if hasattr(sys.stdout, 'reconfigure'):
 sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("organizer.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("google").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
colorama_init()


########################################################################
# Configuration
########################################################################


@dataclass
class OrganizerConfig:
    supported_extensions: tuple[str, ...] = (
        '.txt', '.md', '.doc', '.docx', '.pdf'
    )
    analysis_max_tokens: int = 8192
    nomenclature_max_tokens: int = 8192


def load_config(config_path: str = "config.json") -> OrganizerConfig:
    """Load configuration from JSON file or use defaults if not found."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return OrganizerConfig(
                supported_extensions=tuple(data.get(
                    "supported_extensions",
                    OrganizerConfig.supported_extensions,
                )),
                analysis_max_tokens=data.get(
                    "analysis_max_tokens",
                    OrganizerConfig.analysis_max_tokens,
                ),
                nomenclature_max_tokens=data.get(
                    "nomenclature_max_tokens",
                    OrganizerConfig.nomenclature_max_tokens,
                ),
            )
    except FileNotFoundError:
        logger.info("Config file not found. Using defaults.")
        return OrganizerConfig()


config = load_config()
supported_ext = config.supported_extensions
analysis_max_tokens = config.analysis_max_tokens
nomenclature_max_tokens = config.nomenclature_max_tokens

# Maximum number of characters from a file that will be sent to the LLM
MAX_INPUT_CHARS = 100_000

# Default analysis result returned when processing fails or no text is available
EMPTY_ANALYSIS_RESULT = {
    'tags': [],
    'summary': 'No summary available.',
    'entities': {},
    'key_phrases': [],
    'sentiment': 'neutral',
    'document_form': '',
    'document_purpose': ''
}

########################################################################
# LLM Initialization
########################################################################

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not set. Exiting.")
    sys.exit(1)

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
    "thinking_config": types.ThinkingConfig(include_thoughts=True), # This is correct for the client.models.generate_content call
    "response_mime_type": "application/json",
    # Note: When using generate_content_stream, this entire dict will be passed
    # to the 'config' parameter.
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

def save_analysis(file_hash: str, analysis: dict, filepath: str) -> None:
    """Save analysis to cache along with metadata."""
    storage = load_storage()
    storage["files"][file_hash] = {
        "analysis": analysis,
        "filepath": filepath,
        "timestamp": time.time(),
    }
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
    def __init__(self, analysis_config, nomenclature_config, api_key: str | None = None):
        self.analysis_config = analysis_config
        self.nomenclature_config = nomenclature_config
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
    
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
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    **self.analysis_config
                )
            )
        except genai.errors.APIError as e:
            logger.error(f"LLM API error during analysis: {e}")
            return EMPTY_ANALYSIS_RESULT
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}")
            return EMPTY_ANALYSIS_RESULT

        if response and response.text:
            try:
                json_string = response.text.strip()
                if json_string.startswith("```"):
                    json_string = re.sub(r"^```(?:json)?", "", json_string)
                    json_string = json_string.rstrip("`")
                try:
                    result = json.loads(json_string)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", json_string, re.DOTALL)
                    if match:
                        result = json.loads(match.group(0))
                    else:
                        raise
                validate(instance=result, schema=analysis_response_schema)
                return result
            except Exception as e:
                logger.error(f"Failed parsing LLM analysis JSON: {e}")
        return EMPTY_ANALYSIS_RESULT

    def propose_structure(self, analysis_results: list[dict], user_feedback: str) -> dict:
        """Propose a folder structure using the nomenclature model."""
        # Prompt for the nomenclature model. With ThinkingConfig enabled, the model
        # will provide its reasoning as 'thought' parts before the main response.
        prompt = (
            "You have a list of files with detailed metadata including tags, summary, entities, key phrases, sentiment, document form, and document purpose. "
            "Your job: reorganize the files based on their content and propose a neat and logical folder structure. "
            "Group files by their thematic similarities, key phrases, and as a possible second layer on document form/purpose. "
            "You may use a flat structure or a hierarchical folder structure. If some groups are subsets of others, create nested folders. If not, keep it simpler. "
            f"Consider this user feedback to improve your proposal:\\n{user_feedback}\\n\\n"
            "The files are:\n\n"
            + json.dumps(analysis_results, indent=2) + "\n\n"
            "Return only a JSON object: folder names as keys, and values either arrays of file paths "
            "or nested objects for subfolders."
        )

        logger.info(f"Sending to LLM (nomenclature) with a request for reasoning.")

        response_text = ""
        try:
            stream = self.client.models.generate_content_stream(
                model='gemini-2.5-flash-preview-05-20',
                contents=prompt,
                generation_config=types.GenerationConfig(**self.nomenclature_config)
            )

            print(Fore.CYAN + "\nLLM Reasoning Process:" + Style.RESET_ALL)
            for chunk in stream:
                for part in chunk.parts:
                    if hasattr(part, 'thought') and part.thought:
                        print(part.text, end="")
                    else:
                        response_text += part.text
            print(Fore.CYAN + "\n--- End of Reasoning ---" + Style.RESET_ALL)
            
        except genai.errors.APIError as e:
            logger.error(f"LLM API error during structure proposal: {e}")
            return {"Misc": [item["filepath"] for item in analysis_results]}
        except Exception as e:
            logger.error(f"Unexpected error during structure proposal: {e}")
            return {"Misc": [item["filepath"] for item in analysis_results]}

        if response_text:
            try:
                json_string = response_text.strip()
                if json_string.startswith("```json"):
                    json_string = json_string[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()
                
                return json.loads(json_string)
            except json.JSONDecodeError:
                logger.error("Error parsing JSON from LLM for nomenclature proposal.")
                logger.error(f"Received text was: {response_text}")
        else:
            logger.error("No final content response from LLM for nomenclature proposal.")
            
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
        if len(text) > MAX_INPUT_CHARS:
            logger.warning(
                f"Text from {fpath} exceeds {MAX_INPUT_CHARS} characters; trimming"
            )
            text = text[:MAX_INPUT_CHARS]
        if not text.strip():
            analysis = EMPTY_ANALYSIS_RESULT
        else:
            logger.info(f"Sending {fpath} to LLM")
            analysis = self.llm_client.analyze_text(text)

        # Include the filepath in the analysis result
        combined_result = {
            "filepath": fpath,
            **analysis
        }

        # Save analysis for future use
        save_analysis(current_hash, analysis, fpath)
        return combined_result
       

    def process_directory(self, input_dir: str, exclude_dirs: set[str] | None = None) -> list[dict]:
        """Scan directory, process supported files, return analysis results."""
        files = self.scan_directory(input_dir, exclude_dirs=exclude_dirs)
        analysis_results = []
        for fpath in tqdm(files, desc="Analyzing files"):
            analysis = self.process_file(fpath)
            analysis_results.append({"filepath": fpath, "analysis": analysis})
        return analysis_results

    @staticmethod
    def scan_directory(directory: str, *, exclude_dirs: set[str] | None = None) -> list[str]:
        """Return a list of supported files in a directory, skipping excluded dirs."""
        exclude_dirs = {os.path.abspath(p) for p in (exclude_dirs or set())}
        found = []
        for root, _, filenames in os.walk(directory):
            if os.path.abspath(root) in exclude_dirs:
                continue
            for filename in filenames:
                if filename.lower().endswith(supported_ext):
                    found.append(str(Path(root) / filename))
        return found


########################################################################
# File Organization
########################################################################

def organize_files(chosen_structure: dict, output_dir: str, *, dry_run: bool = False) -> None:
    """Recursively create folders and move files according to the proposed structure."""
    for theme, content_ in chosen_structure.items():
        safe_theme = re.sub(r'[<>:"/\\|?*]', '_', theme.strip()).replace(' ', '_')
        theme_dir = Path(output_dir) / Path(safe_theme)
        theme_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(content_, list):
            for f in tqdm(content_, desc=f"{theme_dir}"):
                src = Path(f)
                if src.exists() and not dry_run:
                    dest = theme_dir / src.name
                    if dest.exists():
                        base, ext = os.path.splitext(src.name)
                        count = 1
                        while dest.exists():
                            dest = theme_dir / f"{base}_{count}{ext}"
                            count += 1
                        logger.warning(f"Name collision for {src}; renaming to {dest.name}")
                    shutil.move(str(src), str(dest))
                elif not src.exists():
                    logger.warning(f"{f} does not exist.")
        elif isinstance(content_, dict):
            organize_files(content_, str(theme_dir), dry_run=dry_run)
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

def gather_user_feedback_and_improve(
    llm_client: LLMClient,
    analysis_results: list[dict],
    output_dir: str,
    *,
    auto_approve: bool = False,
    dry_run: bool = False,
):
    """Loop until user approves structure or quits."""
    user_feedback = ""
    while True:
        proposed_structure = llm_client.propose_structure(analysis_results, user_feedback)
        logger.info("Proposed structure:")
        formatted_structure = format_structure_output(proposed_structure)
        print(Fore.CYAN + "\nProposed structure:\n" + Style.RESET_ALL)
        print(formatted_structure)

        if auto_approve:
            if not dry_run:
                organize_files(proposed_structure, output_dir, dry_run=dry_run)
            generate_report(analysis_results, output_dir)
            logger.info("Files organized.")
            break

        # Aggregate and display key metadata for user awareness
        all_authors = set(
            author
            for item in analysis_results
            for author in item["analysis"].get("entities", {}).get("authors", [])
        )
        all_recipients = set(
            recipient
            for item in analysis_results
            for recipient in item["analysis"].get("entities", {}).get("intended_recipients", [])
        )
        all_key_phrases = set(
            phrase
            for item in analysis_results
            for phrase in item["analysis"].get("key_phrases", [])
        )
        overall_sentiment = [item["analysis"].get("sentiment", "neutral") for item in analysis_results]
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
            organize_files(proposed_structure, output_dir, dry_run=dry_run)
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

def run_organizer(
    input_dir: str,
    output_dir: str | None = None,
    *,
    force_reanalyze: bool = False,
    auto_approve: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full organization pipeline."""
    if not os.path.exists(input_dir):
        logger.error(f"Input path {input_dir} does not exist.")
        return

    output_dir = output_dir or os.path.join(os.path.dirname(input_dir), "organized_folder")
    logger.info(f"Output path: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(analysis_generation_config, nomenclature_generation_config)
    analyzer = FileAnalyzer(llm_client, force_reanalyze)

    analysis_results = analyzer.process_directory(input_dir, exclude_dirs={output_dir})
    if not analysis_results:
        logger.info("No supported files found.")
        return

    gather_user_feedback_and_improve(
        llm_client,
        analysis_results,
        output_dir,
        auto_approve=auto_approve,
        dry_run=dry_run,
    )


def clear_cache() -> None:
    """Delete the analysis cache file if it exists."""
    path = Path(STORAGE_FILE)
    if path.exists():
        path.unlink()
        print("Cache cleared.")
    else:
        print("No cache file found.")


def interactive_menu() -> None:
    """Simple text-based menu for common tasks."""
    while True:
        print(Fore.YELLOW + "\nFile Organizer Menu" + Style.RESET_ALL)
        print("1. Organize a directory")
        print("2. Clear analysis cache")
        print("3. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            input_dir = input("Input directory: ").strip()
            output_dir = input(
                "Output directory (blank for default): "
            ).strip() or None
            auto = input("Auto approve? [y/N]: ").strip().lower() == "y"
            dry = input("Dry run? [y/N]: ").strip().lower() == "y"
            run_organizer(
                input_dir,
                output_dir,
                auto_approve=auto,
                dry_run=dry,
            )
        elif choice == "2":
            clear_cache()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Try again.")


########################################################################
# Main Execution Flow
########################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Organize files using LLM.")
    parser.add_argument("input_dir", nargs="?", help="Path to input directory.")
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("--force_reanalyze", action="store_true", help="Force re-analysis of all files.")
    parser.add_argument("--auto_approve", action="store_true", help="Skip prompts and apply the first proposed structure.")
    parser.add_argument("--dry_run", action="store_true", help="Show proposed structure without moving files.")
    parser.add_argument("--menu", action="store_true", help="Launch interactive menu.")
    args = parser.parse_args()

    if args.menu or not args.input_dir:
        interactive_menu()
        return

    run_organizer(
        args.input_dir,
        args.output_dir,
        force_reanalyze=args.force_reanalyze,
        auto_approve=args.auto_approve,
        dry_run=args.dry_run,
    )




if __name__ == "__main__":
    main()
