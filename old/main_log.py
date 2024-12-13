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


# configure the API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set. Exiting.")
    sys.exit(1)
print("API key loaded successfully.")
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

print("Response schema defined.")

# Define the generation configuration
analysis_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 8000,
    "response_schema": analysis_response_schema,
    "response_mime_type": "application/json",
}

print("Analysis generation configuration set.")

# Create the model
analysis_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=analysis_generation_config,
)

print("Analysis model created.")

nomenclature_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 4000,
    "response_mime_type": "application/json",
}

nomenclature_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=nomenclature_generation_config,
)

print("Nomenclature model created.")

def extract_text_from_docx(filepath: str) -> str:
    print(f"Extracting text from DOCX: {filepath}")
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_pdf(filepath: str) -> str:
    print(f"Extracting text from PDF: {filepath}")
    text = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def extract_text_from_doc(filepath: str) -> str:
    print(f"Extracting text from DOC: {filepath}")
    try:
        result = subprocess.run(["antiword", filepath], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Warning: Could not extract text from {filepath} using antiword.")
    except FileNotFoundError:
        print("Warning: antiword not found. Unable to extract text from .doc file.")
    return ""

def extract_file_content(filepath: str) -> str:
    print(f"Extracting content from file: {filepath}")
    lower_ext = Path(filepath).suffix.lower()
    if lower_ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif lower_ext == '.docx':
        return extract_text_from_docx(filepath)
    elif lower_ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif lower_ext == '.doc':
        return extract_text_from_doc(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return ""

def analyze_file_content(filepath: str) -> dict:
    print(f"Analyzing file content: {filepath}")
    file_data = extract_file_content(filepath)

    if not file_data.strip():
        print(f"No extractable text found in file: {filepath}")
        return {
            'tags': [],
            'summary': 'No extractable text.'
        }

    print(f"Sending file data for analysis: {filepath}")
    chat_session = analysis_model.start_chat()
    response = chat_session.send_message(file_data)

    if response and response.text:
        try:
            result = json.loads(response.text)
            print(f"Analysis complete for file: {filepath}")
            return {
                'tags': result.get('tags', []),
                'summary': result.get('summary', '')
            }
        except ValueError:
            print(f"Error parsing JSON response for file: {filepath}")
    else:
        print(f"No response for file: {filepath}")

    return {
        'tags': [],
        'summary': 'No summary available.'
    }


def propose_nomenclature_with_llm(analysis_results: list[dict]) -> dict:
    print("Proposing nomenclature with LLM.")
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
            print("Nomenclature proposal complete.")
            return result
        except ValueError:
            print("Error parsing JSON response for nomenclature proposal.")
    else:
        print("No response for nomenclature proposal.")

    # Fallback if no valid response
    print("Using fallback nomenclature proposal.")
    return {"Misc": [item["filepath"] for item in analysis_results]}


def organize_files(chosen_structure: dict) -> None:
    print("Organizing files into proposed structure.")
    for theme, files in chosen_structure.items():
        theme_dir = Path(theme.strip().replace(" ", "_"))
        if not theme_dir.exists():
            print(f"Creating directory: {theme_dir}")
            theme_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            basename = os.path.basename(f)
            src = Path(f)
            dest = theme_dir / basename
            if src.exists():
                print(f"Moving file: {src} to {dest}")
                shutil.move(str(src), str(dest))
            else:
                print(f"Warning: {f} does not exist and cannot be moved.")


def main():
    print("Starting main process.")
    cwd = Path(os.getcwd())
    supported_ext = ('.txt', '.md', '.doc', '.docx', '.pdf')
    files_to_process = [
        str(cwd / f) for f in os.listdir(cwd)
        if (cwd / f).is_file() and f.lower().endswith(supported_ext)
    ]

    if not files_to_process:
        print("No supported files found.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    # Analyze files
    analysis_results = []
    for fpath in files_to_process:
        print(f"Processing file: {fpath}")
        analysis = analyze_file_content(fpath)
        analysis_results.append({"filepath": fpath, "analysis": analysis})

    # Propose nomenclature with LLM
    proposed_structure = propose_nomenclature_with_llm(analysis_results)
    print("Proposed Structure:")
    print(json.dumps(proposed_structure, indent=2))

    # Validate with the user
    user_input = input("Approve this structure? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Structure not approved. Exiting.")
        return

    # Organize files
    organize_files(proposed_structure)

    # Generate a report
    report_path = cwd / "report.json"
    with report_path.open('w', encoding='utf-8') as rep:
        json.dump({
            "analysis_results": analysis_results,
            "final_structure": proposed_structure
        }, rep, indent=2)
    print(f"Report generated: {report_path}")
    print("Files organized successfully.")

if __name__ == "__main__":
    main()