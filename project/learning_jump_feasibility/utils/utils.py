import glob, os
from typing import Tuple
from torch.nn import Module
from torch import load, device
import importlib.util
from .diffusion.DDPM import DDPM
from .diffusion.CDCD import CDCD

from .config import Config

# Import all models from files in ../models/*
DEFAULT_MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.dirname(__file__).replace("utils", DEFAULT_MODELS_DIR)

class ModelLoader:
    """
    Load model from model configuration file
    """
    def __init__(self, absolute_models_path:str="") -> None:
        self.model = None
        # IMPORT MODELS
        absolute_models_path = DEFAULT_MODEL_PATH if absolute_models_path == "" else absolute_models_path
        self.all_models = {}
        self._import_models(absolute_models_path)

    def _import_models(self, absolute_models_path:str):
        # Create relative path
        module_path = absolute_models_path.replace(os.getcwd(), "")
        # Parse as python file import
        module_path = module_path.replace("/", ".")[1:]
        for p in glob.glob(os.path.join(absolute_models_path, "*.py")):
            filename = os.path.split(p)[1].replace(".py", "")

            # Import models
            module_name = f"{module_path}.{filename}"
            spec = importlib.util.spec_from_file_location(module_name, p)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get all the classes defined in the module
            self.all_models.update(
                {name: cls for name, cls in module.__dict__.items() 
                 if isinstance(cls, type)}
                 )

    def load(self, model_name:str, cfg_model:dict) -> Module:
        try:
            self.model = self.all_models[model_name](**cfg_model)
            self.model.__setattr__("name", model_name)
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model number of trainable parameters:", n_params)
        except Exception as e:
            print(e)
            print("Can't load model", model_name)
        
        return self.model
            
def get_model(cfg:Config=None, state_path:str="") -> Module:
    """
    If cfg is provided:
        - Return a model instance.

    If state_path is provided:
        - Return trained model assuming it is save in logs with its
        config yaml file in the same directory
    """
    model = None

    if state_path != "":

        # Get run config
        run_dir = os.path.split(state_path)[0]
        config_path = glob.glob(run_dir + "/*.yaml") + glob.glob(run_dir + "/*.yml")
        assert len(config_path) > 0, f"Config file not found in {run_dir}"
        cfg = Config(config_path[0])

        model_name = cfg.get_value("model_name")
        cfg_model = cfg.model["PARAMS"]

        # Code version of the run directory
        absolute_models_path = os.path.join(os.path.abspath(run_dir), "models")
        model_loader = ModelLoader(absolute_models_path)

        # Load state
        state = load(state_path, map_location=device('cpu'))
        
        if "DDPM" in state["trainer"]:
            # Model instance
            base_model = model_loader.load(model_name, cfg_model)
            model = DDPM(base_model)
        if "CDCD" in state["trainer"]:
            # Model instance
            base_model = model_loader.load(model_name, cfg_model)
            model = CDCD(base_model)
        else:
            # Model instance
            model = model_loader.load(model_name, cfg_model)
        
        model.load_state_dict(state["state_dict"])
        print("Model state restored at epoch", state["epoch"])
    
    elif cfg != None:
        model_loader = ModelLoader()
        model_name = cfg.get_value("model_name")
        cfg_model = cfg.model["PARAMS"]
        # Model instance
        model = model_loader.load(model_name, cfg_model)

    return model