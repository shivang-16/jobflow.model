import importlib
import logging
import os

loaded_models = {}

def load_model(model_version):
    print(f"Attempting to load model version: {model_version}")
    print(f"Current working directory: {os.getcwd()}")
    
    if model_version in loaded_models:
        print("Model found in cache")
        return loaded_models[model_version]

    try:
        module_name = f"models.{model_version}.model"
        print(f"Full module path: {module_name}")
        print(f"Checking if directory exists: {os.path.exists(os.path.join('models', model_version))}")
        
        model_module = importlib.import_module(module_name)
        loaded_models[model_version] = model_module
        print("Model loaded successfully")
        return model_module
    except ModuleNotFoundError as e:
        print(f"Module not found error: {str(e)}")
        logging.error(f"Module not found: {module_name}")
        return None
    except Exception as e:
        print(f"Other error occurred: {str(e)}")
        return None
