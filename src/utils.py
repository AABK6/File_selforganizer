import os
import subprocess
import hashlib
import logging
from pathlib import Path
import shutil # Added import
import docx
import PyPDF2

logger = logging.getLogger(__name__) # Add logger for utils

########################################################################
# File Extraction Functions
########################################################################

def extract_text_from_txt(filepath: str) -> str:
    """Extract text from a .txt or .md file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e: # Catching generic Exception to log it
        logger.warning(f"Failed to read {filepath}: {e}")
        return ""

def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a .docx file."""
    try:
        doc = docx.Document(filepath)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e: # Catching generic Exception to log it
        logger.warning(f"Failed to extract text from {filepath}: {e}")
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
    except Exception as e: # Catching generic Exception to log it
        logger.warning(f"Failed to extract text from {filepath}: {e}")
        return ""

def extract_text_from_doc(filepath: str) -> str:
    """Extract text from a .doc file using antiword."""
    if not shutil.which("antiword"):
        logger.warning(
            "antiword command not found. .doc file processing will be skipped. "
            "Please install antiword to enable text extraction from .doc files."
        )
        return ""
    try:
        result = subprocess.run(["antiword", filepath], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning(f"antiword failed for {filepath}. Return code: {result.returncode}, Error: {result.stderr}")
            return ""
    except FileNotFoundError:
        logger.warning("antiword not found. Skipping .doc extraction.")
        return ""
    except Exception as e: # Catching generic Exception to log it
        logger.warning(f"Error during .doc extraction for {filepath} using antiword: {e}")
        return ""

def extract_file_content(filepath: str) -> str:
    """Dispatch extraction based on file extension."""
    ext = Path(filepath).suffix.lower()
    # Note: supported_ext is not directly used here, dispatching is based on extension.
    # This assumes that if a file type is passed here, its extraction is supported.
    if ext in ['.txt', '.md']:
        return extract_text_from_txt(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    elif ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.doc':
        return extract_text_from_doc(filepath)
    logger.warning(f"Unsupported file type or error for extension {ext} in {filepath}")
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
