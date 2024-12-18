# AI-Powered File Organizer

This project uses AI (specifically the Google Gemini Pro and potentially Gemini 1.5 Flash models) to automatically organize files within a directory based on their content. It analyzes file content, proposes a folder structure, allows for user feedback and refinement, and then physically moves files into the generated structure.

## Features

-   **Multi-Format Support:** Handles various file types, including `.txt`, `.md`, `.docx`, `.pdf`, and `.doc` using the `textract` library.
-   **Content Analysis:** Leverages the Gemini Pro model to extract key entities, categories, and importance levels from files.
-   **Intelligent Structure Proposal:** Generates a hierarchical folder structure based on the analysis, taking into account user-specified constraints (e.g., maximum depth, grouping preferences).
-   **User Feedback and Refinement:** Prompts the user to approve, reject, or provide feedback on the proposed structure, allowing for iterative improvements via the LLM.
-   **Automated File Organization:** Recursively creates the proposed folder structure and moves files to their appropriate locations.
-   **Persistent Storage:** Stores file content, analysis results, and SHA256 hashes in a `storage.json` file to avoid redundant processing.
-   **Reporting:** Generates a `report.json` file summarizing the analysis and organization process.
-   **Configurable:** Allows customization of model parameters (e.g., `max_tokens`, `temperature`, `top_p`) and supported file extensions through a `config.json` file.
-   **Error Handling:** Implements robust error handling, including specific handling for `FileNotFoundError`, `FileExistsError`, `PermissionError`, and unsupported file types, as well as retries with exponential backoff for LLM API requests.
-   **Progress Indication:** Displays progress bars using `tqdm` for file processing and user feedback stages.

## Prerequisites

-   **Python 3.8+:** Ensure you have Python 3.8 or a later version installed.
-   **Google AI Studio API Key:** Obtain an API key from Google AI Studio and set it as the environment variable `GENAI_API_KEY`.
-   **System Dependencies (for `textract`):** You'll need to install several system-level tools to support various file types. The specific requirements depend on your OS and the file types you intend to process. Refer to the `textract` documentation for detailed instructions. Common dependencies include:
    -   `antiword` (for `.doc` files)
    -   `pdftotext` (for `.pdf` files; often part of the `poppler-utils` package)
    -   `libjpeg-dev`, `zlib1g-dev`, `libpng-dev` (for image processing in some cases)
    -   ...and others as needed.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install required system dependencies** for textract:

    ```bash
    # For Debian/Ubuntu:
    sudo apt-get install antiword poppler-utils libjpeg-dev zlib1g-dev libpng-dev
    # Add other package as needed, depending on your OS and required file types
    ```

## Usage

1. **Set your Google AI Studio API key:**

    ```bash
    export GENAI_API_KEY="your_api_key_here"
    ```

2. **Run the script:**

    ```bash
    python main.py <target_folder>
    ```

    -   Replace `<target_folder>` with the path to the directory containing the files you want to organize.

        Options:
            ```bash
            python main.py <input_directory> [--output_dir <output_directory>] [--force_reanalyze]
            ```

            *   Replace `<input_directory>` with the path to the folder you wish to organize.
            *   Use `--output_dir <output_directory>` to specify an output directory. If omitted, files will be organized into an `organized_folder` within the input directory.
            *   Use `--force_reanalyze` to force the re-analysis of all files, ignoring the cache.

            Examples:
            ```bash
            # Organize files in the "my_folder" directory
            python main.py my_folder

            # Organize files in "my_folder" and place the output in "my_output_folder"
            python main.py my_folder --output_dir my_output_folder

            # Re-analyze files in "my_folder" and then organize
            python main.py my_folder --force_reanalyze
            ```

3. **Follow the prompts:**

    -   The script will analyze the files in the target folder.
    -   It will then present a proposed folder structure.
    -   You will be prompted to:
        -   **Approve (a):** Accept the proposed structure and proceed with file organization.
        -   **Reject (r):** Discard the proposed structure and exit.
        -   **Provide Feedback (f):** Give feedback to the AI to refine the structure. You can iteratively provide feedback until you are satisfied with the proposed organization.

4. **File Organization:**

    -   If you approve the structure, the script will create the folders and move the files accordingly.
    -   A `storage.json` file will be created/updated to store file information and analysis results.
    -   A `report.json` file will be generated with a summary of the process.

## Configuration

You can customize the behavior of the script using the `config.json` file:

```json
{
    "supported_extensions": [".txt", ".md", ".doc", ".docx", ".pdf"],
    "analysis_max_tokens": 1024,
    "nomenclature_max_tokens": 1024,
    "analysis_temperature": 0.7,
    "analysis_top_p": 0.8,
    "nomenclature_temperature": 0.2,
    "nomenclature_top_p": 0.5
}
```

-   **`supported_extensions`:** A list of file extensions to be processed.
-   **`analysis_max_tokens`:** The maximum number of tokens for the LLM to generate during the analysis phase.
-   **`nomenclature_max_tokens`:** The maximum number of tokens for the LLM to generate during the structure proposal phase.
-   **`analysis_temperature`:** Controls the randomness of the LLM's output during analysis (higher values are more random).
-   **`analysis_top_p`:** An alternative to temperature that controls the diversity of the LLM's output during analysis (higher values are more diverse).
-   **`nomenclature_temperature`:** Controls the randomness of the LLM's output during structure proposal.
-   **`nomenclature_top_p`:** Controls the diversity of the LLM's output during structure proposal.

## Project Structure

-   **`main.py`:** The main entry point for the script. Handles argument parsing, file processing, LLM interaction, user feedback, and file organization.
-   **`config.py`:** Loads configuration from `config.json` and sets up logging.
-   **`config.json`:** Contains configurable parameters for the script.
-   **`cli.py`:** Defines the command-line interface and the user feedback loop.
-   **`file_utils.py`:** Provides functions for extracting text from files using `textract` and calculating SHA256 file hashes.
-   **`llm_client.py`:** Defines the `LLMClient` class, which interacts with the Gemini Pro/Flash models, handles prompts, and performs response validation.
-   **`organizer.py`:** Contains functions for creating folders recursively and moving files based on the proposed structure.
-   **`requirements.txt`:** Lists the required Python packages.
-   **`storage.json`:** (Generated) Stores file content, analysis results, and hashes.
-   **`report.json`:** (Generated) Provides a summary of the file analysis and organization.
-   **`organizer.log`:** (Generated) Log file for debugging and tracking the script's execution.

## Troubleshooting

-   **`textract` Errors:** If you encounter errors related to `textract`, ensure that you have installed all the necessary system dependencies for the file types you are using. Refer to the `textract` documentation for your specific operating system.
-   **API Key Issues:** Make sure your `GENAI_API_KEY` environment variable is correctly set and that your API key is valid.
-   **LLM Errors:** If the LLM returns invalid JSON or encounters other errors, check the `organizer.log` file for details. You may need to adjust the prompts or model parameters.
-   **Permission Errors:** If you get `PermissionError` during file moving, ensure that you have the necessary permissions to write to the target directory and its subfolders.

## Limitations

-   **LLM Accuracy:** The quality of the file organization depends heavily on the accuracy and capabilities of the underlying language model. The current prompts are designed for general use, but you may need to fine-tune them or experiment with different model parameters to achieve optimal results for your specific files and desired structure.
-   **Complex Structures:** The script is currently designed to handle relatively simple folder structures (up to 3 levels deep). Extremely complex or deeply nested structures might require further adjustments to the prompts and logic.
-   **Ambiguous Content:** Files with very ambiguous or overlapping content might be difficult to categorize accurately, even with user feedback.

## Future Enhancements

-   **Interactive Mode:** Implement a more interactive mode where users can preview the proposed structure and make manual adjustments before the files are moved.
-   **Advanced Prompt Engineering:** Explore more advanced prompt engineering techniques, such as chain-of-thought reasoning, to improve the LLM's ability to generate accurate and complex folder structures.
-   **Model Fine-tuning:** Investigate the possibility of fine-tuning the language model on a dataset of file organization examples to improve performance on specific types of files or desired structures.
-   **Alternative LLMs:** Add support for other large language models, potentially allowing users to choose the model that best suits their needs.
-   **GUI:** Develop a graphical user interface to make the tool more user-friendly.




## Contributions

Contributions are welcome! If you have any ideas for improvements, or find bugs, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.