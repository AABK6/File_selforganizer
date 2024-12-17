import json
import jsonschema
import os

from config import analysis_max_tokens, nomenclature_max_tokens, logger
from genai.credentials import Credentials
from genai.model import Model, GenerateTextParams


analysis_response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "category": {"type": "string"},
            "importance": {"type": "string"},
        },
        "required": ["entity", "category", "importance"],
    },
}

analysis_generation_config = {
    "max_tokens_to_sample": analysis_max_tokens,
    "model": "google/gemini-pro",  # Using Gemini Pro instead of Gemini Flash
    "temperature": 0.5,
    "top_p": 0.9,
}

nomenclature_generation_config = {
    "max_tokens_to_sample": nomenclature_max_tokens,
    "model": "google/gemini-pro",  # Using Gemini Pro instead of Claude-2
    "temperature": 0.0,
    "top_p": 0.9,
}


class LLMClient:
    def __init__(self, analysis_config=analysis_generation_config, nomenclature_config=nomenclature_generation_config):
        self.credentials = Credentials(os.environ.get("GENAI_API_KEY"))  # Using GENAI_API_KEY
        self.model = Model("google/gemini-pro", credentials=self.credentials)  # Using Gemini Pro
        self.analysis_config = analysis_config
        self.nomenclature_config = nomenclature_config

    def analyze_text(self, text):
        prompt = f"""Analyze the following document and extract key entities, categories, and their importance levels, and provide the results in JSON format according to the provided schema: {json.dumps(analysis_response_schema)}

        ## Document:
        {text}"""
        response = self.model.generate_text(GenerateTextParams(**self.analysis_config, prompt=prompt))
        try:
            analysis_result = json.loads(response.text)
            jsonschema.validate(instance=analysis_result, schema=analysis_response_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logger.error(f"Invalid JSON response from LLM: {e}, Response: {response.text}")
            raise
        return analysis_result

    def propose_structure(self, analysis_results, user_feedback=""):
        prompt = f"""Based on the analysis results and user feedback, propose a file organization structure.

        Analysis Results: {analysis_results}
        User Feedback: {user_feedback}"""  
        response = self.model.generate_text(GenerateTextParams(**self.nomenclature_config, prompt=prompt))
        try:
            nomenclature_proposal = response.text
        except Exception as e:
            logger.error(f"Error proposing structure: {e}, Response: {response.text}")
            raise
        return nomenclature_proposal