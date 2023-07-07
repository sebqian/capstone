"""Read configuration file for experiment."""
from typing import Any, Dict
from etils import epath
import yaml


def read_experiment_config(config_file: epath.Path) -> Dict[str, Any]:
    """Read experiment configurations."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config
