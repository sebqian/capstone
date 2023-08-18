"""Read configuration file for experiment."""
from typing import Any, Dict
from pathlib import Path
import yaml


def read_configuration(config_file: Path) -> Dict[str, Any]:
    """Read experiment configurations."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config
