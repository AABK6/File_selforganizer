import json
import jsonschema
import os
import time
import logging # Added
from pathlib import Path
from google import genai
from google.genai import types

# Get current script directory
script_dir = Path(__file__).parent
config_path = script_dir / "config.json"

logger = logging.getLogger(__name__) # Added logger

def load_llm_config():
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"{config_path} not found. Using default LLM parameters.") # Changed to logger.warning
        return {
            "analysis_max_tokens": 8192,
            "nomenclature_max_tokens": 8192,
            "analysis_temperature": 0.7, # Default from main.py analysis_generation_config
            "analysis_top_p": 0.9, # Default from main.py analysis_generation_config
            "nomenclature_temperature": 0.8, # Default from main.py nomenclature_generation_config
            "nomenclature_top_p": 0.9, # Default from main.py nomenclature_generation_config
            "llm_model_analysis": "gemini-1.5-flash-8b", # Default from main.py
            "llm_model_nomenclature": "gemini-1.5-flash-8b"  # Default from main.py
        }

llm_config = load_llm_config()

# Response schema for validation
analysis_response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "category": {"type": "string"},
            "importance": {"type": "string", "enum": ["high", "medium", "low"]},
            "explanation": {"type": "string"},
        },
        "required": ["entity", "category", "importance", "explanation"],
    },
}

class LLMClient:
    def __init__(self, genai_models_client, config): # Pass genai.Client().models and loaded config
        self.models_client = genai_models_client # This is the google.genai.Client().models instance
        self.config = config
        self.analysis_model = self.config.get("llm_model_analysis", 'gemini-1.5-flash')
        self.nomenclature_model = self.config.get("llm_model_nomenclature", 'gemini-1.5-flash')

    def _generate_with_retry(self, model, contents, generation_config, max_retries=3):
        """Generates text with exponential backoff retry mechanism."""
        retries = 0
        while retries < max_retries:
            try:
                response = self.models_client.generate_content( # Changed
                    model=model,
                    contents=contents,
                    config=generation_config
                )
                return response
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning( # Changed to logger.warning
                    f"LLM request failed (attempt {retries}/{max_retries}): {e}. Retrying in {wait_time} seconds."
                )
                time.sleep(wait_time)
        raise Exception(
            f"LLM request failed after {max_retries} retries."
        )

    def analyze_text(self, text):
        """Analyzes text using the LLM and validates the response."""
        prompt = f"""Analyze the following document and extract key entities, their categories, and importance levels. Provide reasoning for each assignment.

        **Constraints:**
        - Classify importance as 'high', 'medium', or 'low'.
        - Provide a concise explanation for each classification.
        - Return the results in JSON format strictly adhering to the following schema:

        ```json
        {json.dumps(analysis_response_schema)}
        ```

        ## Document:
        {text}"""

        # Correctly define generation_config as a dictionary for GenerateContentConfig
        generation_config_dict = {
            "candidate_count": 1,
            "stop_sequences": ["\n\n\n"],
            "max_output_tokens": self.config.get("analysis_max_tokens"),
            "temperature": self.config.get("analysis_temperature"),
            "top_p": self.config.get("analysis_top_p"),
            "response_schema": analysis_response_schema, # This is key for structured output
            "response_mime_type": "application/json" # Ensures JSON output
        }
        # Create the GenerateContentConfig object
        generate_content_config = types.GenerateContentConfig(**generation_config_dict)

        response = self._generate_with_retry(
            model=self.analysis_model,
            contents=[{"parts": [{"text": prompt}]}],
            generation_config=generate_content_config, # Use the updated config object
        )

        response_text = "No response text captured" # Default in case response is None
        try:
            if not response:
                raise ValueError("LLM response object is None.")
            response_text = response.text
            # The response from a schema-enforced model call should be directly parsable JSON
            analysis_result = json.loads(response_text)
            jsonschema.validate(instance=analysis_result, schema=analysis_response_schema)
            return analysis_result
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logger.error( # Changed to logger.error
                f"Invalid JSON response from LLM for analysis: {e}. Response text: '{response_text}'"
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM analysis response processing: {e}") # Changed to logger.error
            if response:
                logger.debug(f"Raw LLM response text: {response.text}") # Changed to logger.debug
            else:
                logger.error("LLM response object was None.") # Changed to logger.error
            raise


    def propose_structure(self, analysis_results, history=""):
        """Proposes a file organization structure using the LLM."""
        # Example-based prompt (few-shot learning) - Examples updated to reflect new structure.
        examples = """
        **Example 1:**
        Analysis Results (list of objects, each with 'filepath' and 'analysis' list):
        [
          {{
            "filepath": "docs/Project_Alpha_Proposal.docx",
            "analysis": [
              {{"entity": "Project Alpha Proposal", "category": "Project Proposals", "importance": "high", "explanation": "Initial proposal for a major project."}},
              {{"entity": "Q4 Goals", "category": "Strategy", "importance": "medium", "explanation": "Describes quarterly goals."}}
            ]
          }},
          {{
            "filepath": "notes/Meeting_Notes_2023-10-26.txt",
            "analysis": [
              {{"entity": "Meeting Notes 2023-10-26", "category": "Meeting Notes", "importance": "medium", "explanation": "Notes from a regular team meeting."}}
            ]
          }}
        ]
        Proposed Structure:
        {{
          "Project_Proposals": {{
            "docs/Project_Alpha_Proposal.docx": ""
          }},
          "Meeting_Notes": {{
            "notes/Meeting_Notes_2023-10-26.txt": ""
          }}
        }}
        """
        # Updated prompt to reflect the new analysis_results structure
        prompt = f"""{examples}

        You have a list of files. For each file, you are given its 'filepath' and an 'analysis' list.
        The 'analysis' list contains extracted entities, where each entity has an 'entity' (name), 'category', 'importance' (high/medium/low), and an 'explanation'.
        Your job: reorganize the files based on this entity information and propose a neat and logical folder structure.

        **Constraints:**
        - Group files by their thematic similarities, primarily using entity categories and importance.
        - Create a maximum of 3 levels of subfolders.
        - Avoid creating folders with only one file unless its entities indicate high importance.
        - Provide the structure in a valid JSON format. Folder names are keys, and values are either arrays of file paths (strings) or nested objects for subfolders.
        - Ensure all original filepaths from the input are present in the output structure.

        Consider this user feedback to improve your proposal:
        {user_feedback}

        The files and their extracted entities are:
        {json.dumps(analysis_results, indent=2)}

        Return only a JSON object representing the folder structure. No extra text or explanations.
        """

        # Correctly define generation_config as a dictionary for GenerateContentConfig
        generation_config_dict = {
            "candidate_count": 1,
            "stop_sequences": ["\n\n\n"],
            "max_output_tokens": self.config.get("nomenclature_max_tokens"),
            "temperature": self.config.get("nomenclature_temperature"),
            "top_p": self.config.get("nomenclature_top_p"),
            "response_mime_type": "application/json" # Ensures JSON output
        }
        # Create the GenerateContentConfig object
        generate_content_config = types.GenerateContentConfig(**generation_config_dict)

        response = self._generate_with_retry(
            model=self.nomenclature_model,
            contents=[{"parts": [{"text": prompt}]}], # Ensure contents is a list of Parts
            generation_config=generate_content_config, # Use the updated config object
        )
        
        response_text = "No response text captured" # Default in case response is None
        try:
            if not response:
                raise ValueError("LLM response object is None for proposing structure.")
            response_text = response.text
            # Remove markdown code block formatting if present
            # Handles ```json\n...\n``` or ```\n...\n```
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            nomenclature_proposal = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM for nomenclature proposal: {e}. Response text: '{response_text}'") # Changed to logger.error
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM nomenclature proposal: {e}") # Changed to logger.error
            if response :
                logger.debug(f"Raw LLM response text for nomenclature: {response.text}") # Changed to logger.debug
            else:
                logger.error("LLM response object was None for nomenclature.") # Changed to logger.error
            raise

        return nomenclature_proposal