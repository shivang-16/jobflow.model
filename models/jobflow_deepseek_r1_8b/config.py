import os

HF_TOKEN = os.getenv('HF_TOKEN')  # Load from environment variable
MODEL_NAME = os.getenv('MODEL_NAME', "shivang-16/JobFlow-DeepSeek-R1-Distill-Llama-8B")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")
