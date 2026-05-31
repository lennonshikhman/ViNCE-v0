from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import sample_index
from .teacher import TeacherModel
from .utils import config_hash, ensure_dir


def _cache_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"{split}_features.npz"


def save_feature_cache(out_dir: Path, split: str, data: dict) -> None:
    ensure_dir(out_dir)
    np.savez_compressed(_cache_path(out_dir, split), **data)


def extract_or_load_features(
    dataset,
    split: str,
    teacher: TeacherModel,
    out_dir: Path,
    config: dict,
    batch_size: int,
    num_workers: int,
    class_names: list[str],
) -> dict:
    ensure_dir(out_dir)
    path = _cache_path(out_dir, split)
    h = config_hash(config)
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if str(cached.get("config_hash", "")) == h:
            return {k: cached[k] for k in cached.files}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features, labels, raw_logits, raw_preds = [], [], [], []
    for images, y in tqdm(loader, desc=f"features {split}"):
        batch_logits = teacher.predict_logits(images)
        batch_features = teacher.extract_features(images)
        features.append(batch_features.numpy())
        raw_logits.append(batch_logits.numpy())
        raw_preds.append(batch_logits.argmax(dim=1).numpy())
        labels.append(torch.as_tensor(y).numpy())
    sample_indices = np.array([sample_index(dataset, i) for i in range(len(dataset))], dtype=np.int64)
    data = {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "raw_imagenet_logits": np.concatenate(raw_logits),
        "raw_imagenet_preds": np.concatenate(raw_preds),
        # These are populated with dataset-label predictions after fitting the
        # frozen-feature teacher head in the pipeline. Keeping placeholders here
        # makes partially extracted caches self-describing if a run is interrupted.
        "teacher_logits": np.concatenate(raw_logits),
        "teacher_preds": np.concatenate(raw_preds),
        "sample_indices": sample_indices,
        "class_names": np.array(class_names, dtype=object),
        "config_hash": np.array(h),
    }
    save_feature_cache(out_dir, split, data)
    return data
