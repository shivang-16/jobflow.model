from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

hf_token = os.getenv('HF_TOKEN')

# Initialize Flask app
app = Flask(__name__)

# Authenticate with Hugging Face Hub
login(hf_token)

# Model details
model_name = "shivang-16/JobFlow-Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded successfully!")

# Function to generate response
def generate_response(question):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a job search assistant, Job-GPT, with deep expertise in job market analytics, resume screening, and structured data extraction from natural language queries. Your task is to analyze the following job query, identify key parameters such as job title, job type, location, salary range, experience requirements, and any other relevant details, and then generate a structured JSON query that can be used to fetch matching job listings from a database.

### Question:
{question}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# API Route
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Missing question parameter"}), 400
    
    response = generate_response(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
