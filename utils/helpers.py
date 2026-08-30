"""Filesystem utility functions."""
import json
from pathlib import Path
from typing import Union


def load_json_config(path: Union[str, Path]) -> dict:
    """Load a JSON config file, failing loudly and clearly if it's missing.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content as a dict.

    Raises:
        FileNotFoundError: If the file doesn't exist — with a clear message
            naming exactly what's missing, rather than a downstream
            JSONDecodeError from an accidentally auto-created empty file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required config file not found: {path}. Create it before running the pipeline."
        )
    with open(path) as f:
        return json.load(f)

def ensure_path(path: Union[str, Path], is_file: bool = False) -> Path:
    """Check whether a path exists, creating it if it doesn't.

    Args:
        path: The path to check/create.
        is_file: If True, treats `path` as a file — creates its parent
            directory (if missing) and an empty file at that path if it
            doesn't already exist. If False (default), treats `path` as a
            directory and creates it (including any missing parents).

    Returns:
        The resolved Path object.
    """
    path = Path(path)

    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path
