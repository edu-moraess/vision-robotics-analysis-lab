from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.arqtech import ModelRegistry, ExperimentLog, describe_architecture
from src.arqtech.detector_interface import ClassicalBackend, ArqtechBackend, select_backend
from src.arqtech.registry import ModelStatus

def test_describe_architecture():
    d = describe_architecture()
    assert d["trained"] is False

def test_registry_baseline_active():
    reg = ModelRegistry(root=str(Path(tempfile.mkdtemp())))
    models = reg.list_models()
    assert any(m["version"] == "classical-cv-baseline" for m in models)

def test_arqtech_backend_unavailable():
    b = ArqtechBackend()
    assert b.is_available() is False
    try:
        b.detect(np.zeros((32, 32, 3), dtype=np.uint8))
        assert False
    except RuntimeError:
        pass

def test_select_backend_falls_back():
    assert isinstance(select_backend("arqtech"), ClassicalBackend)

def test_classical_detects():
    assert isinstance(ClassicalBackend().detect(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)), list)

def test_experiment_log():
    log = ExperimentLog(root=str(Path(tempfile.mkdtemp())))
    rec = log.create(title="Smoke plan")
    assert rec.status == "PLANNED" and log.list_experiments()


def test_arqtech_modular_model_forward_and_heads():
    import torch
    from src.arqtech import ARQTECHModel

    model = ARQTECHModel(num_classes=4)
    x = torch.zeros((2, 3, 64, 64))
    logits = model(x)
    outputs = model.forward_all(x)
    assert logits.shape == (2, 4)
    assert outputs["classification_logits"].shape == (2, 4)
    assert outputs["box"].shape == (2, 4)


def test_arqtech_lifecycle_transitions():
    from src.arqtech.status import ModelStatus, can_transition, transition
    assert can_transition(ModelStatus.NOT_TRAINED, ModelStatus.TRAINING)
    assert transition(ModelStatus.TRAINING, ModelStatus.TRAINED) == "TRAINED"
    assert not can_transition(ModelStatus.TRAINED, ModelStatus.ACTIVE)


def test_architecture_reports_pytorch_and_honest_scope():
    from src.arqtech.architecture import describe_architecture
    description = describe_architecture()
    assert description["framework"] == "PyTorch"
    assert description["trained"] is False
    assert "Synthetic patch classification" in description["current_training_scope"]


def test_arqtech_training_produces_experimental_checkpoint(tmp_path):
    from src.arqtech.train import train_arqtech_v01
    result = train_arqtech_v01(epochs=1, batch_size=16, n_samples=64, seed=3, out_dir=str(tmp_path / "models"))
    assert result.status == "TRAINED_EXPERIMENTAL"
    assert result.lifecycle_status == "TRAINED"
    assert result.checkpoint_path
    assert Path(result.checkpoint_path).exists()
    assert len(result.train_loss) == 1 and len(result.val_loss) == 1


def test_arqtech_experimental_mode_falls_back_without_detection_checkpoint():
    from src.core.pipeline import AnalysisPipeline
    result = AnalysisPipeline(perception_mode="ARQTECH_EXPERIMENTAL").run(
        np.zeros((32, 32, 3), dtype=np.uint8), run_planner=False,
    )
    assert result.model_identity["model"] == "CURRENT DETECTOR"
    assert any("ARQTECH: UNAVAILABLE FOR DETECTION" in note for note in result.notes)
