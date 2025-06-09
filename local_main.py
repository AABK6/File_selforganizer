import argparse
import os
import json
from pathlib import Path
import subprocess
import docx
import PyPDF2
from onnx_llm_client import ONNXLLMClient, DEFAULT_MODEL_DIR

def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(path: Path) -> str:
    try:
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception:
        return ""


def read_pdf(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def read_doc(path: Path) -> str:
    try:
        result = subprocess.run(["antiword", str(path)], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        pass
    return ""


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".doc":
        return read_doc(path)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Local file organizer using ONNX LLM")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Path to ONNX model (defaults to AI Toolkit location)"
    )
    args = parser.parse_args()

    client = ONNXLLMClient(str(args.model_dir))

    SUPPORTED_EXT = {".txt", ".md", ".doc", ".docx", ".pdf"}
    analysis = {}
    for file_path in args.input_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXT:
            text = read_file(file_path)
            if text:
                result = client.analyze_text(text)
            else:
                result = {}
            analysis[str(file_path)] = result

    structure = client.propose_structure(analysis)
    with open('report.json', 'w') as f:
        json.dump({"analysis": analysis, "structure": structure}, f, indent=2)

if __name__ == "__main__":
    main()
