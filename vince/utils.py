from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any



def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_hash(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def read_openai_api_key(path: str = "api.txt") -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"OpenAI API key file not found: {key_path}")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"OpenAI API key file is empty after stripping whitespace: {key_path}")
    return key
