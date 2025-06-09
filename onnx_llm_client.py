import os
import json
import onnxruntime_genai as og
from transformers import AutoTokenizer

class ONNXLLMClient:
    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = og.Model(model_dir)

    def _generate(self, prompt: str, max_tokens: int = 1024) -> str:
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=max_tokens, temperature=0.6, top_p=0.95)
        generator = og.Generator(self.model, params)
        input_tokens = self.tokenizer.encode(prompt)
        generator.append_tokens(input_tokens)
        output_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            output_tokens.extend(generator.get_next_tokens())
        return self.tokenizer.decode(output_tokens)

    def analyze_text(self, text: str) -> dict:
        prompt = (
            "Analyze the following document and extract key entities, their "
            "categories and importance. Provide a JSON array with fields: "
            "entity, category, importance, explanation.\n\nDocument:\n" + text
        )
        result = self._generate(prompt)
        return json.loads(result)

    def propose_structure(self, analysis_results: dict) -> dict:
        prompt = (
            "Using the provided analysis results, propose a folder structure "
            "grouping files by category and importance. Return a JSON object "
            "where keys are folder names and values are lists of files.\n\n" +
            json.dumps(analysis_results)
        )
        result = self._generate(prompt)
        return json.loads(result)
