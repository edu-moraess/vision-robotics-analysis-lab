from .architecture import describe_architecture
from .registry import ModelRegistry
from .status import ModelStatus, can_transition, transition
from .experiment_log import ExperimentLog
from .lifecycle import ARQTECH_LIFECYCLE, LifecycleRecord

try:
    from .model import ARQTECHModel
    from .inference import ARQTECHInference
    from .train import train_arqtech_v01, TrainResult
    from .model_v01 import ARQTechV01, CLASS_NAMES
except Exception:
    ARQTECHModel = None
    ARQTECHInference = None
    train_arqtech_v01 = None
    TrainResult = None
    ARQTechV01 = None
    CLASS_NAMES = ["background", "obstacle", "wall", "person"]

__all__ = [
    "describe_architecture", "ModelRegistry", "ModelStatus",
    "can_transition", "transition", "ExperimentLog", "ARQTECHModel",
    "ARQTECHInference", "train_arqtech_v01", "TrainResult", "ARQTechV01",
    "CLASS_NAMES", "ARQTECH_LIFECYCLE", "LifecycleRecord",
]
