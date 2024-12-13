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
    from google.ai.generativelanguage_v2beta.types import content
except ImportError:
    print("google-generativeai not found. Please install it.")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("organizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.json"):
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


STORAGE_FILE = "analysis_cache.json"

def load_storage() -> dict:
    if Path(STORAGE_FILE).exists():
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}, "nomenclature_comments": {}}

def save_storage(data: dict) -> None:
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_file_id(filepath: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))

def get_cached_analysis(file_id: str) -> dict:
    storage = load_storage()
    return storage["files"].get(file_id)

def save_analysis(file_id: str, analysis: dict, file_hash: str) -> None:
    storage = load_storage()
    storage["files"][file_id] = {"analysis": analysis, "hash": file_hash}
    save_storage(storage)

def save_nomenclature_comment(proposed_structure: dict, comment: str):
    storage = load_storage()
    storage["nomenclature_comments"] = {"structure": proposed_structure, "comment": comment}
    save_storage(storage)

def get_nomenclature_comment():
    storage = load_storage()
    return storage.get("nomenclature_comments", {})


def extract_text_from_txt(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except:
        logger.warning(f"Failed to read {filepath}")
        return ""

def extract_text_from_docx(filepath: str) -> str:
    try:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs)
    except:
        logger.warning(f"Failed to extract text from {filepath}")
        return ""

def extract_text_from_pdf(filepath: str) -> str:
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
    # Requires antiword installed
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


def get_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return ""
    return hasher.hexdigest()


def analyze_file_content(filepath: str, force_analyze: bool = False) -> dict:
    file_id = get_file_id(filepath)
    if not force_analyze:
        cached_data = get_cached_analysis(file_id)
        if cached_data:
            cached_hash = cached_data.get("hash", "")
            current_hash = get_file_hash(filepath)
            if cached_hash == current_hash:
                return cached_data.get("analysis", {})

    file_data = extract_file_content(filepath)
    if not file_data.strip():
        analysis = {'tags': [], 'summary': 'No extractable text.'}
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
                logger.error(f"Error parsing JSON from LLM for {filepath}")
                analysis = {'tags': [], 'summary': 'No summary available.'}
        else:
            logger.error(f"No response from LLM for {filepath}")
            analysis = {'tags': [], 'summary': 'No summary available.'}

    file_hash = get_file_hash(filepath)
    save_analysis(file_id, analysis, file_hash)
    return analysis


def propose_nomenclature_with_llm(analysis_results: list[dict], user_feedback: str = "") -> dict:
    prompt = (
        "You have a set of files, each with tags and summaries. "
        "Your goal: propose the best possible folder structure. "
        "Decide whether a flat structure or a hierarchical one is better. "
        "Group them logically by theme. If hierarchical makes sense, nest folders. "
        "If flat is better, don't force hierarchy. "
        "In any case, produce a JSON that might contain nested objects if hierarchical. "
        "No extra commentary, just the JSON. "
        "User feedback (if any) that should be considered for improving this structure:\n"
        f"{user_feedback}\n"
        "Here are the files:\n\n"
        + json.dumps(analysis_results, indent=2) + "\n\n"
        "Now respond with a single JSON object representing the folder structure."
    )

    chat_session = nomenclature_model.start_chat()
    response = chat_session.send_message(prompt)
    if response and response.text:
        try:
            return json.loads(response.text)
        except ValueError:
            logger.error("Error parsing JSON from LLM for nomenclature proposal.")
    else:
        logger.error("No response from LLM for nomenclature proposal.")
    return {"Misc": [item["filepath"] for item in analysis_results]}


def scan_directory(directory: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(supported_ext):
                files.append(str(Path(root) / filename))
    return files


def organize_files(chosen_structure: dict, output_dir: str) -> None:
    for theme, content_ in chosen_structure.items():
        theme_dir = Path(output_dir) / Path(theme.strip().replace(" ", "_"))
        theme_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(content_, list):
            for f in content_:
                src = Path(f)
                if src.exists():
                    shutil.move(str(src), str(theme_dir / src.name))
                else:
                    logger.warning(f"{f} does not exist.")
        elif isinstance(content_, dict):
            organize_files(content_, str(theme_dir))
        else:
            logger.warning(f"Unexpected structure type for {theme}")


def main():
    parser = argparse.ArgumentParser(description="Organize files using LLM.")
    parser.add_argument("input_dir", type=str, help="Path to input directory.")
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("--force_reanalyze", action="store_true", help="Force re-analysis of all files.")

    args = parser.parse_args()
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        logger.error(f"Input path {input_dir} does not exist.")
        sys.exit(1)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_dir), 'organized_folder')
    logger.info(f"Output path: {output_dir}")

    files_to_process = scan_directory(input_dir)
    if not files_to_process:
        logger.info("No supported files found.")
        return

    analysis_results = []
    for fpath in tqdm(files_to_process, desc="Analyzing files"):
        analysis = analyze_file_content(fpath, force_analyze=args.force_reanalyze)
        analysis_results.append({"filepath": fpath, "analysis": analysis})

    user_feedback = ""
    while True:
        proposed_structure = propose_nomenclature_with_llm(analysis_results, user_feedback)
        logger.info("Proposed structure:")
        logger.info(json.dumps(proposed_structure, indent=2))

        choice = input("Approve, Reject, or Comment? (a/r/c): ").strip().lower()
        if choice == 'a':
            # Approved
            user_comment = input("Optional comment about the final structure (press Enter to skip): ")
            save_nomenclature_comment(proposed_structure, user_comment)
            organize_files(proposed_structure, output_dir)
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
            logger.info("Files organized.")
            break
        elif choice == 'r':
            # Rejected, ask again without changes
            logger.info("User rejected the structure. Trying again with no additional feedback.")
            user_feedback = ""
            continue
        elif choice == 'c':
            # User wants to add a comment/feedback
            user_feedback = input("Enter your feedback to improve the structure: ")
            logger.info("User provided feedback. Will re-propose structure incorporating feedback.")
            continue
        else:
            logger.info("Invalid choice. Please enter 'a', 'r', or 'c'.")


if __name__ == "__main__":
    main()
