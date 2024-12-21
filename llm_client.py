import logging
import os
import json
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from config import logger, config, analysis_max_tokens, nomenclature_max_tokens, analysis_temperature, analysis_top_p, nomenclature_temperature, nomenclature_top_p

class LLMClient:
    def __init__(self):
        try:
            self.client = genai.Client(api_key=os.environ['GENAI_API_KEY'])
        except KeyError:
            logger.error("GOOGLE_API_KEY environment variable not set.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google AI client: {e}")
            raise
        self.analysis_model = 'gemini-2.0-flash-exp'  # Or any model suitable for analysis
        self.nomenclature_model = 'gemini-2.0-flash-exp'  # Or any model suitable for nomenclature

    def analyze_text(self, text_content: str) -> str:
        """Analyzes the text content using the LLM to understand its topic."""
        
        prompt = f"""Analyze the following document and provide a structured analysis in JSON format.
        Focus on extracting key information including tags, summary, entities, key phrases, sentiment, and document metadata.
        Format the output exactly as specified, ensuring dates are in YYYY-MM-DD format.
        
        Document:
        {text_content}
        
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

        class DocumentAnalysis(BaseModel):
            tags: list[str]
            summary: str
            entities: dict = Field(
                ...,
                description="Extracted entities from the document",
                json_schema_extra={
                    "properties": {
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "intended_recipients": {"type": "array", "items": {"type": "string"}},
                        "organizations": {"type": "array", "items": {"type": "string"}},
                        "locations": {"type": "array", "items": {"type": "string"}},
                        "dates": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["authors", "intended_recipients", "organizations", "locations", "dates"]
                }
            )
            key_phrases: list[str]
            sentiment: str = Field(..., enum=["positive", "negative", "neutral"])
            document_form: str
            document_purpose: str
        
        try:
            response = self.client.models.generate_content(
                model=self.analysis_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=analysis_temperature,
                    top_p=analysis_top_p,
                    max_output_tokens=analysis_max_tokens,
                    response_mime_type='application/json',
                    response_schema=DocumentAnalysis
                )
            )
            logger.debug(f"LLM response for analysis:\n{response}")
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return ""

    def propose_structure(self, analysis_results: list[str], feedback: str = None) -> dict:
        """Proposes a folder structure based on the analysis results."""
        prompt ="Based on the following text analysis, propose a hierarchical folder structure to organize these documents, including the specific filenames that should be placed in each folder:\n"
        for i, result in enumerate(analysis_results):
            filename = result.get('filename', f'Document {i+1}')
            prompt += f"Document {i+1} (Filename: {filename}): {result}\n"

        if feedback:
            prompt += f"\nConsidering the feedback: '{feedback}', refine the folder structure."

        prompt += "\nThe folder structure should be a JSON object where keys are folder names and values are either sub-folders (as nested JSON objects) or a list of filenames (strings) if the folder contains files. The structure should specify the exact placement of each file within the folder hierarchy."

        logger.debug(f"Proposing structure with model '{self.nomenclature_model}':\n{prompt}")
        try:
            response = self.client.models.generate_content(
                model=self.nomenclature_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=nomenclature_temperature,
                    top_p=nomenclature_top_p,
                    max_output_tokens=nomenclature_max_tokens,
                    response_mime_type='application/json'
                )
            )
            logger.debug(f"LLM response for structure proposal:\n{response.text}")
            try:
                # Attempt to parse the JSON response
                return json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode JSON response for folder structure: {e}. Raw response: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Error proposing folder structure: {e}")
            return {}