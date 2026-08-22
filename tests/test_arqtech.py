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
