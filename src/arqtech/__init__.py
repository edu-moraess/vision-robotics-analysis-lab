"""ARQTECH experimental package."""
from .architecture import describe_architecture
from .registry import ModelRegistry
from .experiment_log import ExperimentLog
try:
    from .train import train_arqtech_v01, TrainResult
    from .model_v01 import ARQTechV01, CLASS_NAMES
except Exception:
    train_arqtech_v01 = None
    TrainResult = None
    ARQTechV01 = None
    CLASS_NAMES = ["background", "obstacle", "wall", "person"]
__all__ = ["describe_architecture", "ModelRegistry", "ExperimentLog",
           "train_arqtech_v01", "TrainResult", "ARQTechV01", "CLASS_NAMES"]
