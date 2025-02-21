import importlib

loaded_models = {}

def load_model(model_version):
    """Dynamically load a model based on version"""
    if model_version in loaded_models:
        return loaded_models[model_version]

    try:
        module_name = f"models.{model_version}.model"
        model_module = importlib.import_module(module_name)
        loaded_models[model_version] = model_module
        return model_module
    except ModuleNotFoundError:
        return None
