from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve data/output paths relative to project root
    for key, rel in cfg["data"].items():
        cfg["data"][key] = str(_PROJECT_ROOT / rel)
    for key, rel in cfg["outputs"].items():
        cfg["outputs"][key] = str(_PROJECT_ROOT / rel)
    return cfg


def get_project_root() -> Path:
    return _PROJECT_ROOT
