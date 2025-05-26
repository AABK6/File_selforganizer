import onnxruntime_genai as og
import numpy as np
from transformers import AutoTokenizer
import time
import os

# --- Configuration ---
# Replace with the actual path where the AI Toolkit downloaded the model files
model_dir = r"C:\Users\aabec\.aitk\models\DeepSeek\qnn-deepseek-r1-distill-qwen-7b"
# model_dir = r"C:\Users\aabec\.aitk\models\DeepSeek\qnn-deepseek-r1-distill-qwen-14b"
# model_dir = r"c:\Users\aabec\.aitk\models\DeepSeek\DeepSeek-R1-Distilled-NPU-Optimized"

# Check if model directory exists
if not os.path.isdir(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}. Please update 'model_dir'.")

# --- Load Tokenizer ---
print(f"Loading tokenizer from: {model_dir}")
try:
    # trust_remote_code=True might be needed for some custom tokenizer implementations
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Ensure tokenizer files (tokenizer.json, etc.) are present in the model directory.")
    print("Check README for specific tokenizer requirements if issues persist.")
    exit()

# --- Configure QNN Execution Provider Options (Optional - for reference) ---
# Although not passed directly to the Generator constructor in this version,
# these are the typical options you might use if the API changes or for other ORT sessions.
print("Configuring QNN Execution Provider options (for reference)...")
qnn_provider_options = {
    'backend_path': 'QnnHtp.dll', # Specifies the Hexagon Tensor Processor backend
    'htp_performance_mode': 'burst', # Options: burst, high_performance, balanced, etc. [9]
    # 'htp_graph_finalization_optimization_mode': '3', # Example advanced option, consult QNN EP docs if needed
    # 'soc_model': '0', # Example option, consult QNN EP docs
    # 'rpc_control_latency': '100', # Example option
    # --- Precision Note ---
    # 'session.enable_htp_fp16_precision': '1' # Use '1' ONLY if the NPU model is FP16. [9, 29]
                                             # NPU models from AI Toolkit are often INT4/INT8 QDQ [15, 22]
                                             # Enabling this for a non-FP16 model might cause errors.
                                             # Start without it; add if facing precision-related issues with a known FP16 model.
}
print(f"Reference QNN EP options: {qnn_provider_options}")

# --- Load Model using onnxruntime-genai ---
print(f"\nLoading model from directory: {model_dir}")
start_time = time.time()
try:
    # Create the model object using only the model directory path
    og_model = og.Model(model_dir)
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading model with onnxruntime-genai: {e}")
    print("Troubleshooting tips:")
    print("- Verify 'onnxruntime-genai' and 'onnxruntime-qnn' (ARM64) are installed.")
    print("- Ensure NPU drivers are up-to-date.")
    print("- Check if the model directory path is correct and contains all necessary.onnx and config files.")
    print("- Confirm the model structure is compatible with onnxruntime-genai.")
    print("- Try loading with CPU EP first by creating the Generator without provider options.")
    exit()

# --- Input Preparation ---
prompt = "Explain the concept of Neural Processing Units (NPUs) in simple terms."
print(f"\nPreparing input for prompt: '{prompt}'")

# Tokenize the input prompt
# The README recommends avoiding system prompts for DeepSeek-R1 models.[30]
input_tokens = tokenizer.encode(prompt) # Returns a list of token IDs

# --- Set Generation Parameters ---
print("\nSetting generation parameters...")
params = og.GeneratorParams(og_model)
params.set_search_options(max_length=2000, temperature=0.6, top_p=0.95) # Using recommended temp [28]

# --- Generate Text using onnxruntime-genai ---
print("\nGenerating text with onnxruntime-genai (attempting QNN EP implicitly)...")
start_time = time.time()
try:
    # Create the generator object without explicit provider options.
    # onnxruntime-genai might implicitly use QNN EP if onnxruntime-qnn is installed
    # and the model is compatible.
    generator = og.Generator(og_model, params)
    print("Generator created.")

    # Append the input tokens (list) to the generator
    generator.append_tokens(input_tokens)
    print("Input tokens appended to generator.")

    print(f"Prompt: {prompt}")
    print("Generated Text:")
    print("-" * 30)
    # Loop through the generation process
    while not generator.is_done():
        generator.generate_next_token() # Computes logits and selects next token
        new_token = generator.get_next_tokens() # Get the token for the first sequence
        print(tokenizer.decode(new_token), end='', flush=True)

    print("\n" + "-" * 30)
    print(f"Generation finished in {time.time() - start_time:.2f} seconds.")

    # Optional: Get the full generated sequence
    # output_sequence = generator.get_output_sequences()
    # print("\nFull output sequence (decoded):")
    # print(tokenizer.decode(output_sequence))

except Exception as e:
    print(f"\nError during text generation: {e}")
    print("Check model compatibility, input data, and generation parameters.")
    print("If the error relates to incompatible constructor arguments for Generator,")
    print("ensure you are only passing the Model and GeneratorParams objects.")
    print("If the error is 'QNN execution provider is not supported in this build',")
    print("ensure 'onnxruntime-genai' was installed correctly and supports QNN EP.")
    print("Try reinstalling 'onnxruntime-qnn' and 'onnxruntime-genai' in your ARM64 venv.")
