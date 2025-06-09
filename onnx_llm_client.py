import os
import json
import onnxruntime_genai as og
from transformers import AutoTokenizer

# Default model directory used on Copilot+ PCs with the AI Toolkit
DEFAULT_MODEL_DIR = r"C:\Users\aabec\.aitk\models\DeepSeek\qnn-deepseek-r1-distill-qwen-7b"

class ONNXLLMClient:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """Load the tokenizer and model from ``model_dir``.

        If ``model_dir`` is ``None`` it defaults to ``DEFAULT_MODEL_DIR``.
        When the installed build of ``onnxruntime-genai`` does not support the
        QNN execution provider, the constructor falls back to CPU by setting the
        ``ORT_DISABLE_QNN`` environment variable and reloading the model.
        """
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        try:
            self.model = og.Model(model_dir)
        except RuntimeError as e:
            if "QNN execution provider is not supported" in str(e):
                os.environ["ORT_DISABLE_QNN"] = "1"
                self.model = og.Model(model_dir)
            else:
                raise

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
