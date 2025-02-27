from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from . import config

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
is_initialized = False

def initialize_model():
    global model, tokenizer, device, is_initialized
    
    if is_initialized:
        return
        
    try:
        # Authenticate
        login(config.HF_TOKEN)
    
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        is_initialized = True
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def generate_response(question):
    global model, tokenizer, device
    
    if not is_initialized:
        initialize_model()
        
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a job search assistant, Job-GPT, with deep expertise in job market analytics, resume screening, and structured data extraction from natural language queries. Your task is to analyze the following job query, identify key parameters such as job title, job type, location, salary range, experience requirements, and any other relevant details, and then generate a structured JSON query that can be used to fetch matching job listings from a database.

### Question:
{question}

### Response:
"""
    try:
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
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return str(e)
