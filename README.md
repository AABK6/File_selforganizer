# LLM-Powered File Organizer

This project automatically sorts your documents into thematic folders using a language model. Files are scanned and analyzed with Google's Gemini API or an offline ONNX model, and a proposed folder structure is presented for approval. Cached results prevent repeated analysis of unchanged files.

## Features

- **Intelligent analysis** for `.txt`, `.md`, `.doc`, `.docx` and `.pdf` files
- **Colorized, interactive prompts** with optional automatic approval
- **Progress bars** during analysis and file moves
- **Dry-run mode** to preview the proposed structure
- **Configurable** via a `config.json` file
- **Offline ONNX mode** when internet access is unavailable

## Prerequisites

- Python 3.7 or newer
- A Google Cloud project with the Gemini API enabled
- `google-genai`, `python-docx`, `PyPDF2`, `tqdm`, `jsonschema`, and `colorama`

## Setup

1. Install the required packages:
   ```bash
   pip install google-genai python-docx PyPDF2 tqdm jsonschema colorama
   ```
2. Export your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="YOUR_API_KEY"
   ```
3. (Optional) Create a `config.json` to override defaults:
   ```json
   {
       "supported_extensions": [".txt", ".md", ".doc", ".docx", ".pdf"],
       "analysis_max_tokens": 8000,
       "nomenclature_max_tokens": 4000
   }
   ```

## Usage

Run the script with:
```bash
python main.py <input_directory> [--output_dir <dir>] [--force_reanalyze] [--auto_approve] [--dry_run]
```

- If `--output_dir` is omitted, files are organized into `organized_folder` inside the input directory.
- `--force_reanalyze` ignores cached results.
- `--auto_approve` skips prompts and applies the first proposed structure.
- `--dry_run` shows the proposal without moving files.
- `--menu` launches an interactive menu to select these options at runtime.

### Example

```bash
python main.py my_folder --auto_approve
```

To start an interactive session instead, run:

```bash
python main.py --menu
```

After execution the script:
1. Scans for supported files
2. Analyzes their contents
3. Proposes a folder structure for approval
4. Moves files unless `--dry_run` is set
5. Writes `report.json` summarizing the run

## Output Files

- Organized files in the new folder structure
- `report.json` with analysis and final layout
- `analysis_cache.json` storing hashes and cached responses
- `organizer.log` with detailed logs

## Running Locally with ONNX

You can operate entirely offline using an ONNX model. Install the additional packages and supply the model directory:
```bash
pip install onnxruntime-genai transformers
python local_main.py <input_directory> --model_dir /path/to/model
```
The runtime uses the QNN execution provider when available and falls back to CPU otherwise.

## Contributing

Contributions are welcome! Open an issue or submit a pull request.

## License

MIT
