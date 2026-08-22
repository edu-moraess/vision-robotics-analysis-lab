"""ARQTECH — Autonomous Robotics Perception Architecture.\n\nScaffold only: not trained. No fabricated metrics.\n"""
from .registry import ModelRegistry, ModelRecord, ModelStatus
from .experiment_log import ExperimentLog, ExperimentRecord
from .architecture import ArqtechConfig, describe_architecture

__all__ = [
    "ModelRegistry", "ModelRecord", "ModelStatus",
    "ExperimentLog", "ExperimentRecord",
    "ArqtechConfig", "describe_architecture",
]
