import importlib
import logging
import os
import torch

loaded_models = {}
model_locks = {}

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def load_model(model_version):
    print(f"Attempting to load model version: {model_version}")
    
    # Return cached model if available
    if model_version in loaded_models and loaded_models[model_version]:
        print("Using cached model")
        return loaded_models[model_version]

    try:
        module_name = f"models.{model_version}.model"
        if model_version not in model_locks:
            model_module = importlib.import_module(module_name)
            
            # Initialize model if not already initialized
            if not hasattr(model_module, 'is_initialized'):
                print("Initializing model for the first time")
                model_module.initialize_model()
                model_module.is_initialized = True
            
            loaded_models[model_version] = model_module
            print("Model loaded and cached successfully")
            
        return loaded_models[model_version]
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        logging.error(f"Model loading error: {str(e)}")
        return None
