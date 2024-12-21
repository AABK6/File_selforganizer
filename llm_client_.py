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
document_analysis_schema = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20
        },
        "summary": {"type": "string"},
        "entities": {
            "type": "object",
            "properties": {
                "authors": {"type": "array", "items": {"type": "string"}},
                "intended_recipients": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "locations": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "dates": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["authors", "intended_recipients", "organizations", "locations", "dates"]
        },
        "key_phrases": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "document_form": {"type": "string"},
        "document_purpose": {"type": "string"}
    },
    "required": ["tags", "summary", "entities", "key_phrases", "sentiment", "document_form", "document_purpose"]
}

class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable GENAI_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)
        self.analysis_model = 'gemini-1.5-flash'
        self.nomenclature_model = 'gemini-2.0-flash-exp'


    def analyze_text(self, text):
        """Analyzes document content and returns structured information."""
        prompt = f"""Analyze the following document and provide a structured analysis in JSON format.
        Focus on extracting key information including tags, summary, entities, key phrases, sentiment, and document metadata.
        Format the output exactly as specified, ensuring dates are in YYYY-MM-DD format.
        
        Document:
        {text}
        
        Provide analysis in the following JSON structure:
        {{
            "tags": ["limit to 20 most relevant categorical labels"],
            "summary": "brief document overview",
            "entities": {{
                "authors": ["detected author names"],
                "intended_recipients": ["detected recipient names"],
                "organizations": ["max 10 key organizations"],
                "locations": ["max 10 key locations"],
                "dates": ["dates in YYYY-MM-DD format"]
            }},
            "key_phrases": ["max 15 important phrases"],
            "sentiment": "positive/negative/neutral",
            "document_form": "document type",
            "document_purpose": "document intent"
        }}"""

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=analysis_max_tokens,
            temperature=analysis_temperature,
            top_p=analysis_top_p
        )

        response = self._generate_with_retry(
            model=self.analysis_model,
            contents=prompt,
            config=generation_config
        )

        if response:
            try:
                analysis = json.loads(response.text)
                jsonschema.validate(analysis, document_analysis_schema)
                return analysis
            except (json.JSONDecodeError, jsonschema.exceptions.ValidationError) as e:
                logger.error(f"Error processing analysis response: {str(e)}")
                raise ValueError("Failed to generate valid document analysis")
        return None
    

    def _generate_with_retry(self, model, contents, config, max_retries=3):
        """Generates text with exponential backoff retry mechanism."""
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                retries += 1
                wait_time = 1 ** retries
                logger.warning(
                    f"LLM request failed (attempt {retries}/{max_retries}): {e}. Retrying in {wait_time} seconds."
                )
                time.sleep(wait_time)
        raise Exception(
            f"LLM request failed after {max_retries} retries."
        )





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
            contents=prompt,  # Correct format for contents
            config=generation_config,
        )
        try:
            nomenclature_proposal = response.text
        except Exception as e:
            logger.error(f"Error proposing structure: {e}, Response: {response.text}")
            raise

        return nomenclature_proposal