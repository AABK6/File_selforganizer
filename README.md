Okay, here's a `README.md` file for your project, designed to be clear, informative, and helpful for users:

```markdown
# LLM-Powered File Organizer

This Python script organizes files in a directory (including subdirectories) using the Gemini API for content analysis and folder structure recommendations. It leverages cached analysis results to avoid redundant API calls, and provides a user-friendly way to reorganize files based on thematic content.

## Features

*   **Intelligent File Analysis:** Uses the Gemini API to extract tags and summaries from text-based files (`.txt`, `.md`, `.doc`, `.docx`, `.pdf`).
*   **Thematic Folder Creation:** Proposes a folder structure based on the thematic similarity of analyzed files using the Gemini API.
*   **Persistent Caching:** Stores analysis results and file hashes to avoid re-analyzing unchanged files in subsequent runs.
*   **User Feedback:** Allows users to review and approve the proposed folder structure, and add comments for record.
*   **Configurable Settings:** Allows to customize the maximum number of tokens used in the analysis and nomenclature models, and the list of supported extensions
*   **Detailed Logging:** Provides both console and file logging to track the progress and troubleshoot issues.
*   **Command-Line Interface:** Offers flexible command-line arguments for input and output directory specification, and for forcing file re-analysis.
*  **Automatic Output Path:** If you omit the option for the output folder, then the files will be moved to the `organized_folder` folder in the same path as the input folder.

## Prerequisites

*   Python 3.7 or higher.
*   A Google Cloud project with the Gemini API enabled.
*   The `google-generativeai` library, `python-docx`, `PyPDF2`, `tqdm`, `uuid`, `hashlib`, and `argparse` libraries.

## Setup

1.  **Install Required Libraries:**

    ```bash
    pip install google-generativeai python-docx PyPDF2 tqdm uuid hashlib argparse
    ```

2.  **Set the Gemini API Key:**

    Set your `GEMINI_API_KEY` environment variable:

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

    Replace `YOUR_API_KEY` with your actual Gemini API key.

3.  **Create a `config.json` (Optional):**

    Create a `config.json` file in the same directory as the script to customize settings such as:
    *   `supported_extensions`: A list of file extensions to process (e.g., `[".txt", ".md", ".doc", ".docx", ".pdf"]`).
    *    `analysis_max_tokens`: The maximum number of tokens for the analysis model (e.g., `8000`).
    *    `nomenclature_max_tokens`: The maximum number of tokens for the nomenclature model (e.g., `4000`).
    *   If this file is not created, default values will be used.

    Example `config.json`:

    ```json
    {
        "supported_extensions": [".txt", ".md", ".doc", ".docx", ".pdf"],
        "analysis_max_tokens": 8000,
        "nomenclature_max_tokens": 4000
    }
    ```

## Usage

Run the script from your terminal using the following command:

```bash
python main.py <input_directory> [--output_dir <output_directory>] [--force_reanalyze]
```

*   Replace `<input_directory>` with the path to the folder you wish to organize.
*   Use `--output_dir <output_directory>` to specify an output directory. If omitted, files will be organized into an `organized_folder` within the input directory.
*   Use `--force_reanalyze` to force the re-analysis of all files, ignoring the cache.

### Examples

```bash
# Organize files in the "my_folder" directory
python main.py my_folder

# Organize files in "my_folder" and place the output in "my_output_folder"
python main.py my_folder --output_dir my_output_folder

# Re-analyze files in "my_folder" and then organize
python main.py my_folder --force_reanalyze
```

After the script runs, it will:

1.  **Scan the directory** for supported files.
2.  **Analyze the file contents** (using cached data if available, or by using the Gemini API).
3.  **Propose a folder structure** based on the analyzed content.
4.  **Prompt for approval**.
5.  **Organize files** into the proposed folder structure based on their content.
6. **Generate a report**. It will be a JSON file that contains the analysis results, final folder structure and nomenclature comments.

## Output

*   **Organized files:** Files will be moved into new folders according to the proposed structure.
*   **`report.json`:** A JSON file containing the analysis results, file IDs, final folder structure and user's comments.
*   **`analysis_cache.json`:** A JSON file used to cache the analysis results, and the file hashes.
*   **`organizer.log`:** A text file containing the logs of the operations made during the script execution, and also the output in the console.

## Important Notes

*   The script uses file hashes to avoid re-analyzing unchanged files, which reduces costs and time.
*   Make sure that you've set up the Gemini API correctly, and that you have enough credits on the Google Cloud platform to perform the analysis.
*   The script moves the files instead of copying them, which should preserve disk space and allow you to organize more data.
*   The script generates the output inside the specified `output_dir` folder.
*   The script is designed to be used on the directory level. If you pass it a single file, the behavior may be different from what's intended.
*   The user is able to provide a comment on the proposed nomenclature before the files are moved.

## Supported File Types

The script currently supports:

*   `.txt`
*   `.md`
*   `.doc`
*   `.docx`
*   `.pdf`

This list can be customized in the `config.json` file.

## Contributions

Contributions are welcome! If you have any ideas for improvements, or find bugs, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
```

This `README.md` should be a good starting point. It covers the main features, prerequisites, usage instructions, and other important information to help users understand your project. Feel free to customize it further based on any specific details you want to highlight.

