from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DATASET_NAMES = ["cifar10", "cats_vs_dogs", "oxford_iiit_pet", "flowers102", "food101"]


@dataclass
class DatasetConfig:
    name: str
    max_train_samples: int
    max_test_samples: int
    max_val_samples: Optional[int] = None


@dataclass
class ViNCEConfig:
    seed: int = 123
    datasets_dir: Path = Path("datasets")
    artifacts_dir: Path = Path("artifacts")
    device: str = "auto"  # "auto" uses CUDA when available, otherwise CPU.
    batch_size: int = 64
    num_workers: int = 2

    teacher_model: str = "resnet18_frozen_features_linear_probe"
    teacher_head_max_iter: int = 1000
    teacher_head_solver: str = "lbfgs"
    teacher_head_c: float = 1.0
    pca_components: int = 16
    tree_max_depth: int = 4
    tree_min_samples_leaf: int = 20
    target_mode: str = "teacher_hard"  # "teacher_hard" or "ground_truth"

    generate_descriptors: bool = False
    openai_model: str = "gpt-4.1-mini"
    descriptor_confidence_threshold: float = 0.6
    max_images_per_side: int = 4
    max_nodes_to_describe: int = 10

    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=lambda: {
        "cifar10": DatasetConfig("cifar10", 2000, 500),
        "cats_vs_dogs": DatasetConfig("cats_vs_dogs", 2000, 500),
        "oxford_iiit_pet": DatasetConfig("oxford_iiit_pet", 2000, 500),
        "flowers102": DatasetConfig("flowers102", 2000, 500, 500),
        "food101": DatasetConfig("food101", 3000, 750),
    })

    def dataset_names(self) -> List[str]:
        return list(DATASET_NAMES)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["datasets_dir"] = str(self.datasets_dir)
        data["artifacts_dir"] = str(self.artifacts_dir)
        return data


def default_config() -> ViNCEConfig:
    return ViNCEConfig()
