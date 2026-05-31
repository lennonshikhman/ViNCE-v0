from __future__ import annotations

import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

from .utils import ensure_dir, write_json

CATS_DOGS_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    class_names: list[str]
    val: Optional[Dataset] = None


def _class_names(ds: Dataset) -> list[str]:
    base = ds.dataset if isinstance(ds, Subset) else ds
    if hasattr(base, "classes"):
        return list(base.classes)
    if hasattr(base, "_labels"):
        labels = sorted(set(base._labels))
        return [str(x) for x in labels]
    if hasattr(base, "datasets") and base.datasets:
        return _class_names(base.datasets[0])
    return []


def _subset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, sorted(indices[:max_samples]))


def sample_index(dataset: Dataset, local_index: int) -> int:
    if isinstance(dataset, Subset):
        return int(dataset.indices[local_index])
    return int(local_index)


def _resolve_concat(dataset: Dataset, index: int) -> tuple[Dataset, int]:
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative"):
        for child, end in zip(dataset.datasets, dataset.cumulative):
            start = end - len(child)
            if index < end:
                return child, index - start
    return dataset, index


def sample_path(dataset: Dataset, local_index: int) -> Optional[Path]:
    idx = sample_index(dataset, local_index)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    base, idx = _resolve_concat(base, idx)
    if hasattr(base, "samples"):
        return Path(base.samples[idx][0])
    if hasattr(base, "_image_files"):
        return Path(base._image_files[idx])
    return None


def sample_pil_image(dataset: Dataset, local_index: int) -> Optional[Image.Image]:
    idx = sample_index(dataset, local_index)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    base, idx = _resolve_concat(base, idx)
    path = sample_path(base, idx)
    if path and path.exists():
        return Image.open(path).convert("RGB")
    if hasattr(base, "data"):
        return Image.fromarray(base.data[idx]).convert("RGB")
    return None


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def prepare_cats_vs_dogs(root: Path, seed: int, train_fraction: float = 0.8, force: bool = False) -> None:
    zip_path = root / "raw" / "kagglecatsanddogs_5340.zip"
    extracted = root / "extracted"
    cleaned = root / "cleaned"
    manifest_path = cleaned / "manifest.json"
    if manifest_path.exists() and not force:
        return
    download_file(CATS_DOGS_URL, zip_path)
    if force and extracted.exists():
        shutil.rmtree(extracted)
    if not (extracted / "PetImages").exists():
        ensure_dir(extracted)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)
    if force and cleaned.exists():
        shutil.rmtree(cleaned)
    for split in ["train", "test"]:
        for cls in ["cat", "dog"]:
            ensure_dir(cleaned / split / cls)

    rng = random.Random(seed)
    manifest = {"kept": [], "skipped": [], "seed": seed, "train_fraction": train_fraction}
    for source_name, class_name in [("Cat", "cat"), ("Dog", "dog")]:
        files = sorted((extracted / "PetImages" / source_name).glob("*.jpg"))
        rng.shuffle(files)
        for src in tqdm(files, desc=f"clean {class_name}"):
            try:
                with Image.open(src) as img:
                    img.verify()
                with Image.open(src) as img:
                    rgb = img.convert("RGB")
                    split = "train" if rng.random() < train_fraction else "test"
                    dst = cleaned / split / class_name / src.name
                    rgb.save(dst, format="JPEG")
                manifest["kept"].append({"source": str(src), "dest": str(dst), "split": split, "class": class_name})
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                manifest["skipped"].append({"source": str(src), "error": str(exc)})
    write_json(manifest_path, manifest)


def prepare_dataset(name: str, datasets_dir: Path, seed: int, force: bool = False) -> None:
    ensure_dir(datasets_dir)
    if name == "cats_vs_dogs":
        prepare_cats_vs_dogs(datasets_dir / "cats_vs_dogs", seed=seed, force=force)
        return
    # torchvision datasets download lazily in load_dataset; this function keeps run_all readable.


def load_dataset(
    name: str,
    datasets_dir: Path,
    transform: Callable,
    max_train: int,
    max_test: int,
    seed: int,
    max_val: Optional[int] = None,
) -> DatasetBundle:
    val = None
    if name == "cifar10":
        root = datasets_dir / "cifar10"
        train = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        test = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    elif name == "cats_vs_dogs":
        prepare_cats_vs_dogs(datasets_dir / "cats_vs_dogs", seed=seed)
        root = datasets_dir / "cats_vs_dogs" / "cleaned"
        train = datasets.ImageFolder(root / "train", transform=transform)
        test = datasets.ImageFolder(root / "test", transform=transform)
    elif name == "oxford_iiit_pet":
        root = datasets_dir / "oxford_iiit_pet"
        train = datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", transform=transform, download=True)
        test = datasets.OxfordIIITPet(root=root, split="test", target_types="category", transform=transform, download=True)
    elif name == "flowers102":
        root = datasets_dir / "flowers102"
        train = datasets.Flowers102(root=root, split="train", transform=transform, download=True)
        val = datasets.Flowers102(root=root, split="val", transform=transform, download=True)
        test = datasets.Flowers102(root=root, split="test", transform=transform, download=True)
    elif name == "food101":
        root = datasets_dir / "food101"
        train = datasets.Food101(root=root, split="train", transform=transform, download=True)
        test = datasets.Food101(root=root, split="test", transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    train = _subset(train, max_train, seed)
    val = _subset(val, max_val, seed + 2) if val is not None else None
    test = _subset(test, max_test, seed + 1)
    return DatasetBundle(train=train, val=val, test=test, class_names=_class_names(train))


class torch_concat(Dataset):
    def __init__(self, datasets_: Sequence[Dataset]):
        self.datasets = list(datasets_)
        self.cumulative = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumulative.append(total)
        self.classes = _class_names(self.datasets[0])

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int):
        for ds, end in zip(self.datasets, self.cumulative):
            start = end - len(ds)
            if index < end:
                return ds[index - start]
        raise IndexError(index)
