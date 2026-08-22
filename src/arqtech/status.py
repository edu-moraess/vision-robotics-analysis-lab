from __future__ import annotations

from enum import Enum


class ModelStatus(str, Enum):
    NOT_TRAINED = "NOT TRAINED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"
    # Backward-compatible registry values used by older experiments.
    PLANNED = "PLANNED"
    DEPRECATED = "DEPRECATED"


_ALLOWED = {
    ModelStatus.NOT_TRAINED: {ModelStatus.TRAINING, ModelStatus.ARCHIVED},
    ModelStatus.TRAINING: {ModelStatus.TRAINED, ModelStatus.FAILED},
    ModelStatus.TRAINED: {ModelStatus.VALIDATING, ModelStatus.ARCHIVED},
    ModelStatus.VALIDATING: {ModelStatus.VALIDATED, ModelStatus.FAILED},
    ModelStatus.VALIDATED: {ModelStatus.ACTIVE, ModelStatus.ARCHIVED},
    ModelStatus.ACTIVE: {ModelStatus.ARCHIVED},
    ModelStatus.ARCHIVED: set(),
    ModelStatus.FAILED: {ModelStatus.TRAINING, ModelStatus.ARCHIVED},
    ModelStatus.PLANNED: {ModelStatus.TRAINING, ModelStatus.ARCHIVED},
    ModelStatus.DEPRECATED: set(),
}


def can_transition(current: str | ModelStatus, target: str | ModelStatus) -> bool:
    current_status = ModelStatus(current)
    target_status = ModelStatus(target)
    return target_status in _ALLOWED[current_status]


def transition(current: str | ModelStatus, target: str | ModelStatus) -> str:
    current_status = ModelStatus(current)
    target_status = ModelStatus(target)
    if not can_transition(current_status, target_status):
        raise ValueError(f"Invalid model lifecycle transition: {current_status.value} -> {target_status.value}")
    return target_status.value
