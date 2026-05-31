from __future__ import annotations

from vince.config import default_config
from vince.data import load_dataset, prepare_dataset
from torchvision.models import ResNet18_Weights
from vince.utils import ensure_dir, set_seed


def main() -> None:
    config = default_config()
    set_seed(config.seed)
    ensure_dir(config.datasets_dir)
    transform = ResNet18_Weights.DEFAULT.transforms()
    for name in config.dataset_names():
        ds_cfg = config.dataset_configs[name]
        prepare_dataset(name, config.datasets_dir, config.seed)
        load_dataset(name, config.datasets_dir, transform, ds_cfg.max_train_samples, ds_cfg.max_test_samples, config.seed)
        print(f"prepared {name}")


if __name__ == "__main__":
    main()
