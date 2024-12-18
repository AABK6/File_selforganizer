import json
import jsonschema
import os
import time
from config import (
    analysis_max_tokens,
    nomenclature_max_tokens,
    analysis_temperature,
    analysis_top_p,
    nomenclature_temperature,
    nomenclature_top_p,
    logger,
)
from google import genai
from google.genai import types

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
    def __init__(self):
        self.api_key = os.environ.get("GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable GENAI_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)
        self.analysis_model = 'gemini-pro'
        self.nomenclature_model = 'gemini-pro'

    def _generate_with_retry(self, model, contents, generation_config, max_retries=3):
        """Generates text with exponential backoff retry mechanism."""
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generation_config  # Correct parameter name
                )
                return response
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(
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

        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["\n\n\n"],
            max_output_tokens=analysis_max_tokens,
            temperature=analysis_temperature,
            top_p=analysis_top_p
        )

        response = self._generate_with_retry(
            model=self.analysis_model,
            contents=[{"parts": [{"text": prompt}]}],  # Correct format for contents
            generation_config=generation_config,
        )

        try:
            analysis_result = json.loads(response.text)
            jsonschema.validate(instance=analysis_result, schema=analysis_response_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logger.error(
                f"Invalid JSON response from LLM: {e}, Response: {response.text}"
            )
            raise

        return analysis_result

    def propose_structure(self, analysis_results, history=""):
        """Proposes a file organization structure using the LLM."""
        # Example-based prompt (few-shot learning)
        examples = """
        **Example 1:**
        Analysis Results:
        [
            {{"entity": "Project Alpha Proposal", "category": "Project Proposals", "importance": "high", "explanation": "Initial proposal for a major project."}},
            {{"entity": "Meeting Notes 2023-10-26", "category": "Meeting Notes", "importance": "medium", "explanation": "Notes from a regular team meeting."}},
            {{"entity": "Expense Report Q4", "category": "Financial Documents", "importance": "medium", "explanation": "Quarterly expense report."}}
        ]
        Proposed Structure:
        {{
            "Project Proposals": {{
                "Project Alpha Proposal.docx": ""
            }},
            "Meeting Notes": {{
                "Meeting Notes 2023-10-26.txt": ""
            }},
            "Financial Documents": {{
                "Expense Report Q4.pdf": ""
            }}
        }}

        **Example 2:**
        Analysis Results:
        [
            {{"entity": "Invoice #1234", "category": "Invoices", "importance": "high", "explanation": "Invoice for a completed project phase."}},
            {{"entity": "Design Draft v2", "category": "Design Documents", "importance": "high", "explanation": "Latest draft of the product design."}},
            {{"entity": "Team Building Event Photos", "category": "Miscellaneous", "importance": "low", "explanation": "Photos from a non-work related event."}}
        ]
        Proposed Structure:
        {{
            "Invoices": {{
                "Invoice #1234.pdf": ""
            }},
            "Design Documents": {{
                "Design Draft v2.docx": ""
            }},
            "Miscellaneous": {{
                "Team Building Event Photos.zip": ""
            }}
        }}
        """

        prompt = f"""{examples}

        Based on the provided analysis results, propose a file organization structure. Consider the importance and category of each entity.

        **Constraints:**
        - Create a maximum of 3 levels of subfolders.
        - Group files by category primarily, then by importance if appropriate.
        - Avoid creating folders with only one file unless it has high importance.
        - Provide the structure in a clear, easily understandable format.

        Analysis Results: {analysis_results}
        {history}"""

        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["\n\n\n"],
            max_output_tokens=nomenclature_max_tokens,
            temperature=nomenclature_temperature,
            top_p=nomenclature_top_p
        )

        response = self._generate_with_retry(
            model=self.nomenclature_model,
            contents=[{"parts": [{"text": prompt}]}],  # Correct format for contents
            generation_config=generation_config,
        )
        try:
            nomenclature_proposal = response.text
        except Exception as e:
            logger.error(f"Error proposing structure: {e}, Response: {response.text}")
            raise

        return nomenclature_proposal