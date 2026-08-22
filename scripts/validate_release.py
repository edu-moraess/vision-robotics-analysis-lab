from pathlib import Path
import importlib
import sys
import tomllib

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
with (root / "pyproject.toml").open("rb") as handle:
    project = tomllib.load(handle)
assert project["project"]["name"] == "vision-robotics-analysis-lab"
for module in ("numpy", "torch", "groq", "src.arqtech.v03", "src.ml.metrics"):
    importlib.import_module(module)
requirements = (root / "requirements.txt").read_text(encoding="utf-8")
for dependency in ("numpy", "groq", "opencv-python-headless==5.0.0.93"):
    assert dependency in requirements
assert "opencv-python==" not in requirements
assert "ultralytics>=" not in requirements
print("RELEASE_STATIC_VALIDATION_OK")
